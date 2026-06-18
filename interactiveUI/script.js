import embeddingData from './embeddings.json' with { type: 'json' };
import { melSpecDbToAudio, playFloatPCM, griffinLim, stft, istft } from './AudioProcessing.js';
const melPinv = await fetch('./mel_pinv.json').then(r => r.json());

var canvas = document.getElementById("canvas");
var ctx = canvas.getContext("2d");


canvas.width = window.innerWidth;
canvas.height = window.innerHeight;

const labels = ['African Pied Wagtail',
 'Barn Swallow',
 'Black Woodpecker',
 'Black-headed Gull',
 'Canada Goose',
 'Carrion Crow',
 'Coal Tit',
 'Common Blackbird',
 'Common Chaffinch',
 'Common Chiffchaff',
 'Common Cuckoo',
 'Common House Martin',
 'Common Linnet',
 'Common Moorhen',
 'Common Nightingale',
 'Common Pheasant',
 'Common Redpoll',
 'Common Redshank',
 'Common Redstart',
 'Common Reed Bunting',
 'Common Snipe',
 'Common Starling',
 'Common Swift',
 'Common Whitethroat',
 'Common Wood Pigeon',
 'Corn Bunting',
 'Dunlin',
 'Dunnock',
 'Eurasian Blackcap',
 'Eurasian Blue Tit',
 'Eurasian Bullfinch',
 'Eurasian Collared Dove',
 'Eurasian Coot',
 'Eurasian Golden Oriole',
 'Eurasian Jay',
 'Eurasian Magpie',
 'Eurasian Nuthatch',
 'Eurasian Oystercatcher',
 'Eurasian Reed Warbler',
 'Eurasian Skylark',
 'Eurasian Tree Sparrow',
 'Eurasian Treecreeper',
 'Eurasian Wren',
 'Eurasian Wryneck',
 'European Bee-eater',
 'European Golden Plover',
 'European Goldfinch',
 'European Green Woodpecker',
 'European Greenfinch',
 'European Herring Gull',
 'European Honey Buzzard',
 'European Nightjar',
 'European Robin',
 'European Turtle Dove',
 'Garden Warbler',
 'Goldcrest',
 'Great Spotted Woodpecker',
 'Great Tit',
 'Grey Partridge',
 'Grey Plover',
 'House Sparrow',
 'Lesser Whitethroat',
 'Long-tailed Tit',
 'Marsh Tit',
 'Marsh Warbler',
 'Meadow Pipit',
 'Northern Lapwing',
 'Northern Raven',
 'Red Crossbill',
 'Red-throated Loon',
 'Redwing',
 'River Warbler',
 'Rock Dove',
 'Rook',
 'Sedge Warbler',
 'Song Thrush',
 'Spotted Flycatcher',
 'Stock Dove',
 'Tawny Owl',
 'Tree Pipit',
 'Western Jackdaw',
 'Western Yellow Wagtail',
 'Willow Ptarmigan',
 'Willow Tit',
 'Willow Warbler',
 'Wood Sandpiper',
 'Wood Warbler',
 'Yellowhammer']

var colors = [];
for(var i = 0; i < embeddingData.length; i++){
    colors.push(`hsla(${(labels.indexOf(embeddingData[i].label)/labels.length)*360}, 100%, 50%, 50%)`);
}

var zoom = 0.05;
var panx = 0;
var pany = 0;
var panning = false;
var mouseposx = 0;
var mouseposy = 0;

var minimumaspect = Math.min(canvas.width, canvas.height)
var maximumaspect = Math.max(canvas.width, canvas.height)
const aspectRatio = maximumaspect/minimumaspect;

// set canvas scale and position
ctx.translate(canvas.width/2, canvas.height/2);
ctx.scale(minimumaspect, minimumaspect);
// ctx.scale(canvas.width, canvas.height);

// setInterval(update, 60/1000);

// function update(){
//     plot_latent_space(0.05, 0, 2);
// }
plot_latent_space(zoom, panx, pany);

canvas.addEventListener("wheel", scroll)
canvas.addEventListener("mousedown", mousedown)
canvas.addEventListener("mouseup", mouseup)
canvas.addEventListener("mousemove", mousemove)

function scroll(event){
    const sensitivity = 0.005;
    const oldzoom = zoom;
    zoom *= 1+sensitivity*(event.deltaY);

    const panfactorx = ((mouseposx-canvas.width/2)/minimumaspect)
    const panfactory = ((mouseposy-canvas.height/2)/minimumaspect)

    panx += panfactorx/zoom - panfactorx/oldzoom;
    pany += panfactory/zoom - panfactory/oldzoom;
    plot_latent_space(zoom, panx, pany);

    canvas.style.cursor = 'zoom-in';
}

function mousedown(){
    panning = true;
    canvas.style.cursor = 'move';
}
function mouseup(){
    panning = false;
    canvas.style.cursor = 'default';
}

function mousemove(event){
    if(panning){
        const dx = event.clientX-mouseposx;
        const dy = event.clientY-mouseposy;
        panx+=(dx/zoom)/minimumaspect;
        pany+=(dy/zoom)/minimumaspect;
        plot_latent_space(zoom, panx, pany);
    }

    mouseposx = event.clientX;
    mouseposy = event.clientY;

    canvas.style.cursor = 'default';
}

function plot_latent_space(zoom, panx, pany){

    ctx.fillStyle = "black";
    // ctx.fillRect(0,0,canvas.width,canvas.height);
    ctx.fillRect(-1*aspectRatio, -1*aspectRatio,2*aspectRatio,2*aspectRatio);

    ctx.save();

    ctx.scale(zoom, zoom)
    ctx.translate(panx, pany)
    const fontsize = Math.max(15/(zoom*minimumaspect), 0.05)

    ctx.font = `bold ${fontsize}px Arial`;


    const radius = Math.max(0.005/zoom, 0.02);

    for(var i = 0; i < embeddingData.length; i++){

        const chunkData = embeddingData[i];

        const UMAP = chunkData.UMAP;

        ctx.beginPath();
        ctx.arc(UMAP[0], UMAP[1], radius, 0, 2 * Math.PI);
        ctx.fillStyle = colors[i];
        ctx.fill();

        if(zoom > 0.1){
            ctx.fillStyle = "rgba(255,255,255,0.5)"; 
            ctx.fillText(chunkData.label, UMAP[0]-fontsize*5, UMAP[1]+radius);
        }
    }

    ctx.restore();
}


async function runGenerator() {

    console.log("called")
    // Load only the decoder model file
    const session = await ort.InferenceSession.create('../published models/best_decoder.onnx', {
        executionProviders: ['wasm'],
        externalData: [
            {
                path: 'best_decoder.onnx.data', // This must match the exact filename the error asked for
                data: "../published models/best_decoder.onnx.data"                    // The actual path/URL to download the weights from
            }
        ]
    });


    const decoderInputData = new Float32Array(embeddingData[0].features);
    const inputTensor = new ort.Tensor('float32', decoderInputData, [1, 128]);

    // Use the exact input name you specified in the Python script
    const feeds = { embedding: inputTensor };

    // Run inference
    const results = await session.run(feeds);
    
    // Read the output
    const specData = results.spectrogram.data;
    const nMels = 128;
    const nFrames = specData.length / nMels;

    const { audio, sr } = melSpecDbToAudio(specData, [1, 1, nMels, nFrames], melPinv, {
    sr: 22050,
    nFft: 2048,
    hopLength: 512,
    nIter: 32,
    });

    playFloatPCM(audio, sr);
}

// function playBirdSong(audioFloatArray, sampleRate) {
//   const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  
//   // Create an audio buffer container
//   const buffer = audioCtx.createBuffer(1, audioFloatArray.length, sampleRate);
  
//   // Directly copy the Float32 numbers into the speaker channel
//   buffer.getChannelData(0).set(audioFloatArray);

//   // Connect buffer to hardware speakers and play
//   const source = audioCtx.createBufferSource();
//   source.buffer = buffer;
//   source.connect(audioCtx.destination);
//   source.start();
// }
// async function initPyodide() {
//     try {
//         // 2. Start Pyodide runtime environment
//         pyodideInstance = await loadPyodide();
//         statusText.innerText = "Installing librosa and scipy dependencies...";

//         // 3. Load micropip package manager to fetch external pure Python packages
//         await pyodideInstance.loadPackage(["micropip", "numpy", "scipy"]);
//         const micropip = pyodideInstance.pyvars.micropip;

//         // 4. Install librosa dynamically over the web
//         await pyodideInstance.runPythonAsync(`
//                     import micropip
//                     await micropip.install('librosa')
//                 `);

//         statusText.innerText = "Python environment is fully ready.";
//         playBtn.innerText = "Reconstruct & Play Audio";
//         playBtn.disabled = false;
//     } catch (err) {
//         statusText.innerText = "Error loading Pyodide: " + err.message;
//     }
// }

// async function runAudioReconstruction(reconstruction) {
//     statusText.innerText = "Processing Griffin-Lim phase inversion in Python...";
//     playBtn.disabled = true;

//     // 5. Expose your JavaScript 2D array data globally to the Python runtime scope
//     globalThis.js_spec_db = reconstruction;

//     // 6. Execute your exact Python inversion logic pipeline
//     const rawAudioSamples = await pyodideInstance.runPythonAsync(`
//                 import librosa
//                 import numpy as np
//                 from js import js_spec_db

//                 # Convert JavaScript matrix proxy cleanly into a standard numpy array
//                 spec_db = np.array(js_spec_db.to_py())

//                 # Replicate your exact Python logic pipeline
//                 spec_power = librosa.db_to_power(spec_db)
//                 reconstructed_audio = librosa.feature.inverse.mel_to_audio(
//                     spec_power, 
//                     sr=22050, 
//                     n_fft=2048, 
//                     hop_length=512
//                 )

//                 # Send raw 1D float array back out into the JavaScript scope
//                 reconstructed_audio.tolist()
//             `);

//     // 7. Play the returned sample arrays using the Native Browser Web Audio API
//     playGeneratedAudioBuffer(rawAudioSamples, 22050);
// }

// function playGeneratedAudioBuffer(samplesArray, sampleRate) {
//     statusText.innerText = "Playing reconstructed sound waves!";
//     playBtn.disabled = false;

//     const audioCtx = new (window.AudioContext || window.webkitAudioContext)();

//     // Create a blank 1-channel buffer matching sample length at 22.05kHz 
//     const audioBuffer = audioCtx.createBuffer(1, samplesArray.length, sampleRate);
//     const channelData = audioBuffer.getChannelData(0);

//     // Copy Python numerical array results into Web Audio Float32 slot
//     for (let i = 0; i < samplesArray.length; i++) {
//         channelData[i] = samplesArray[i];
//     }

//     // Route audio context stream to physical browser hardware node
//     const sourceNode = audioCtx.createBufferSource();
//     sourceNode.buffer = audioBuffer;
//     sourceNode.connect(audioCtx.destination);
//     sourceNode.start();
// }