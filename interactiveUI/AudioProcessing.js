/**
 * Mel spectrogram (0-1 normalized dB) -> playable audio, entirely in JS.
 *
 * Pipeline:
 *   1. undoRescale: value*80 - 80                 (your normalization)
 *   2. dbToPower:   10^(db/10)                    (matches librosa.db_to_power)
 *   3. melToLinear: multiply by precomputed mel pseudoinverse
 *   4. griffinLim:  iterative phase reconstruction + ISTFT -> waveform
 *
 * Usage (see bottom of file for a full example):
 *
 *   const melPinv = await fetch('./mel_pinv.json').then(r => r.json());
 *   const { audio, sr } = melSpecDbToAudio(
 *     results.spectrogram.data,
 *     results.spectrogram.dims,
 *     melPinv
 *   );
 *   playFloatPCM(audio, sr);
 *
 * Requires n_fft to be a power of two (2048 is fine).
 */

// ---------------------------------------------------------------------
// FFT: iterative radix-2 Cooley-Tukey, in-place on parallel re/im arrays.
// n must be a power of two.
// ---------------------------------------------------------------------
function fft(re, im) {
  const n = re.length;
  if (n <= 1) return;

  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }

  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wRe = Math.cos(ang);
    const wIm = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let curRe = 1;
      let curIm = 0;
      const half = len / 2;
      for (let k = 0; k < half; k++) {
        const aRe = re[i + k];
        const aIm = im[i + k];
        const bRe = re[i + k + half] * curRe - im[i + k + half] * curIm;
        const bIm = re[i + k + half] * curIm + im[i + k + half] * curRe;
        re[i + k] = aRe + bRe;
        im[i + k] = aIm + bIm;
        re[i + k + half] = aRe - bRe;
        im[i + k + half] = aIm - bIm;
        const nextRe = curRe * wRe - curIm * wIm;
        const nextIm = curRe * wIm + curIm * wRe;
        curRe = nextRe;
        curIm = nextIm;
      }
    }
  }
}

function ifft(re, im) {
  const n = re.length;
  for (let i = 0; i < n; i++) im[i] = -im[i];
  fft(re, im);
  for (let i = 0; i < n; i++) {
    re[i] /= n;
    im[i] = -im[i] / n;
  }
}

// ---------------------------------------------------------------------
// Windowing and padding (matches librosa defaults: periodic Hann window,
// center=True with reflect padding)
// ---------------------------------------------------------------------
function hannWindow(n) {
  const w = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / n);
  }
  return w;
}

function reflectPad(signal, padLen) {
  const n = signal.length;
  const out = new Float64Array(n + 2 * padLen);
  out.set(signal, padLen);
  for (let i = 0; i < padLen; i++) {
    out[padLen - 1 - i] = signal[Math.min(i + 1, n - 1)];
    out[padLen + n + i] = signal[Math.max(n - 2 - i, 0)];
  }
  return out;
}

// ---------------------------------------------------------------------
// STFT / ISTFT. Frames are { re, im } half-spectra of length n_fft/2 + 1
// (real-signal redundancy removed); ISTFT rebuilds full spectrum via
// Hermitian symmetry before the inverse FFT.
// ---------------------------------------------------------------------
function stft(signal, nFft, hopLength, window) {
  const padLen = Math.floor(nFft / 2);
  const padded = reflectPad(signal, padLen);
  const half = nFft / 2;
  const numFrames = 1 + Math.floor((padded.length - nFft) / hopLength);
  const frames = [];

  for (let f = 0; f < numFrames; f++) {
    const re = new Float64Array(nFft);
    const im = new Float64Array(nFft);
    const start = f * hopLength;
    for (let i = 0; i < nFft; i++) {
      re[i] = padded[start + i] * window[i];
    }
    fft(re, im);
    frames.push({ re: re.slice(0, half + 1), im: im.slice(0, half + 1) });
  }
  return frames;
}

function istft(frames, nFft, hopLength, window) {
  const half = nFft / 2;
  const numFrames = frames.length;
  const outLen = nFft + hopLength * (numFrames - 1);
  const signal = new Float64Array(outLen);
  const winSum = new Float64Array(outLen);

  for (let f = 0; f < numFrames; f++) {
    const re = new Float64Array(nFft);
    const im = new Float64Array(nFft);
    re.set(frames[f].re.subarray(0, half + 1));
    im.set(frames[f].im.subarray(0, half + 1));
    for (let k = 1; k < half; k++) {
      re[nFft - k] = re[k];
      im[nFft - k] = -im[k];
    }
    ifft(re, im);

    const start = f * hopLength;
    for (let i = 0; i < nFft; i++) {
      signal[start + i] += re[i] * window[i];
      winSum[start + i] += window[i] * window[i];
    }
  }

  for (let i = 0; i < outLen; i++) {
    if (winSum[i] > 1e-8) signal[i] /= winSum[i];
  }

  const padLen = Math.floor(nFft / 2);
  return signal.slice(padLen, outLen - padLen);
}

// ---------------------------------------------------------------------
// Griffin-Lim with momentum (matches librosa's default algorithm/quality)
// magnitude: array of Float64Array(n_fft/2 + 1), one per time frame
// ---------------------------------------------------------------------
function griffinLim(magnitude, nFft, hopLength, window, nIter = 32, momentum = 0.99) {
  const half = nFft / 2;
  const momentumRatio = momentum / (1 + momentum);

  let angles = magnitude.map(() => {
    const re = new Float64Array(half + 1);
    const im = new Float64Array(half + 1);
    for (let k = 0; k <= half; k++) {
      const a = Math.random() * 2 * Math.PI;
      re[k] = Math.cos(a);
      im[k] = Math.sin(a);
    }
    return { re, im };
  });

  let rebuilt = angles.map(() => ({
    re: new Float64Array(half + 1),
    im: new Float64Array(half + 1),
  }));

  const applyMagnitude = (currentAngles) =>
    magnitude.map((row, f) => {
      const re = new Float64Array(half + 1);
      const im = new Float64Array(half + 1);
      for (let k = 0; k <= half; k++) {
        re[k] = row[k] * currentAngles[f].re[k];
        im[k] = row[k] * currentAngles[f].im[k];
      }
      return { re, im };
    });

  let signal;
  for (let iter = 0; iter < nIter; iter++) {
    const tprev = rebuilt;
    signal = istft(applyMagnitude(angles), nFft, hopLength, window);
    rebuilt = stft(signal, nFft, hopLength, window);

    angles = rebuilt.map((frame, f) => {
      const re = new Float64Array(half + 1);
      const im = new Float64Array(half + 1);
      for (let k = 0; k <= half; k++) {
        re[k] = frame.re[k] - momentumRatio * tprev[f].re[k];
        im[k] = frame.im[k] - momentumRatio * tprev[f].im[k];
        const mag = Math.hypot(re[k], im[k]) + 1e-8;
        re[k] /= mag;
        im[k] /= mag;
      }
      return { re, im };
    });
  }

  return istft(applyMagnitude(angles), nFft, hopLength, window);
}

// ---------------------------------------------------------------------
// Top-level pipeline
// ---------------------------------------------------------------------

/**
 * @param {Float32Array|number[]} specDbData - flat decoder output (0-1 normalized)
 * @param {number[]} dims - tensor dims from ONNX, e.g. [1,1,128,T] or [128,T]
 * @param {{shape: number[], data: number[]}} melPinv - loaded mel_pinv.json
 * @param {object} options
 */
function melSpecDbToAudio(specDbData, dims, melPinv, options = {}) {
  const { sr = 22050, nFft = 2048, hopLength = 512, nIter = 32 } = options;

  const nMels = dims[dims.length - 2];
  const nFrames = dims[dims.length - 1];
  const nFreqBins = nFft / 2 + 1;

  if (melPinv.shape[0] !== nFreqBins || melPinv.shape[1] !== nMels) {
    throw new Error(
      `mel_pinv shape ${melPinv.shape} doesn't match expected [${nFreqBins}, ${nMels}]. ` +
        `Check that n_fft/n_mels match between export_mel_pinv.py and this call.`
    );
  }

  // 1+2: undo normalization, then db -> power. Layout assumed [nMels, nFrames]
  // row-major, matching numpy's reconstruction[0][0] shape (n_mels, T).
  const melPower = new Float64Array(nMels * nFrames);
  for (let i = 0; i < melPower.length; i++) {
    const db = specDbData[i] * 80 - 80;
    melPower[i] = Math.pow(10, db / 10);
  }

  // 3: mel -> linear magnitude via pseudoinverse matmul, per frame
  const pinv = melPinv.data;
  const magnitudeFrames = [];
  for (let t = 0; t < nFrames; t++) {
    const frame = new Float64Array(nFreqBins);
    for (let f = 0; f < nFreqBins; f++) {
      let sum = 0;
      const pinvRowOffset = f * nMels;
      for (let m = 0; m < nMels; m++) {
        sum += pinv[pinvRowOffset + m] * melPower[m * nFrames + t];
      }
      frame[f] = Math.max(sum, 0); // pseudoinverse isn't guaranteed non-negative
    }
    magnitudeFrames.push(frame);
  }

  // 4: Griffin-Lim phase reconstruction + final ISTFT
  const window = hannWindow(nFft);
  const audio = griffinLim(magnitudeFrames, nFft, hopLength, window, nIter);

  return { audio, sr };
}

// ---------------------------------------------------------------------
// Playback
// ---------------------------------------------------------------------
function playFloatPCM(samples, sampleRate) {
  const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  const buffer = audioCtx.createBuffer(1, samples.length, sampleRate);
  buffer.copyToChannel(Float32Array.from(samples), 0);
  const source = audioCtx.createBufferSource();
  source.buffer = buffer;
  source.connect(audioCtx.destination);
  source.start();
  return audioCtx;
}

export { melSpecDbToAudio, playFloatPCM, griffinLim, stft, istft };

/* ----------------------------- Example -------------------------------

import { melSpecDbToAudio, playFloatPCM } from './griffin_lim.js';

async function runGenerator() {
  const session = await ort.InferenceSession.create('../published models/best_decoder.onnx', {
    executionProviders: ['wasm'],
  });

  const decoderInputData = new Float32Array(embeddingData[0].features);
  const inputTensor = new ort.Tensor('float32', decoderInputData, [1, 128]);
  const results = await session.run({ embedding: inputTensor });

  const melPinv = await fetch('./mel_pinv.json').then(r => r.json());
  const { audio, sr } = melSpecDbToAudio(
    results.spectrogram.data,
    results.spectrogram.dims,
    melPinv,
    { nFft: 2048, hopLength: 512, nIter: 32 }
  );

  playFloatPCM(audio, sr);
}

------------------------------------------------------------------------ */