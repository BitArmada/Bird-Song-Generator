import embeddingData from './embeddings.json' with { type: 'json' };
import { melSpecDbToAudio, playFloatPCM, griffinLim, stft, istft } from './AudioProcessing.js';
const melPinv = await fetch('./mel_pinv.json').then(r => r.json());

var canvas = document.getElementById("canvas");
var ctx = canvas.getContext("2d");

canvas.width = window.innerWidth;
canvas.height = window.innerHeight;

// Global state
let mode = 'selecting'; // 'panning' or 'selecting'
let selectedIndices = []; // Array of indices of selected datapoints
let session = null; // ONNX session
let currentAudioData = null; // Current generated audio
let currentAudioSampleRate = null; // Sample rate of current audio
let volume = 1.0; // Volume level (0-1, where 1 = 100% = original volume)
let isPlaying = false; // Whether audio is currently playing
let playbackStartTime = 0; // Timestamp when playback started
let playbackDuration = 0; // Duration of the audio in seconds
let playbackAnimationId = null; // Animation frame ID for progress line
let currentSpectrogramData = null; // Store spectrogram for redrawing with progress line
let currentSpectrogramNMels = 0; // Store nMels for redrawing
let currentSpectrogramNFrames = 0; // Store nFrames for redrawing
let autoScaleVolume = true; // Auto-scale volume based on audio peak

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

var zoom = 0.095569634676;
var panx = -0.6087862155379818;
var pany = -8.336139088914303;
var panning = false;
var mouseposx = 0;
var mouseposy = 0;

var comX = 0;
var comY = 0;

var minimumaspect = Math.min(canvas.width, canvas.height)
var maximumaspect = Math.max(canvas.width, canvas.height)
var aspectRatio = maximumaspect/minimumaspect;

// Touch variables for mobile
var touchDistance = 0;
var lastTouchDistance = 0;
var isTouching = false;

// Function to handle canvas resize
function handleCanvasResize() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    
    minimumaspect = Math.min(canvas.width, canvas.height);
    maximumaspect = Math.max(canvas.width, canvas.height);
    aspectRatio = maximumaspect / minimumaspect;
    
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.translate(canvas.width/2, canvas.height/2);
    ctx.scale(minimumaspect, minimumaspect);
    
    plot_latent_space(zoom, panx, pany);
}

// Initialize model on startup
async function initializeModel() {
    try {
        session = await ort.InferenceSession.create('../published models/best_decoder.onnx', {
            executionProviders: ['wasm'],
            externalData: [
                {
                    path: 'best_decoder.onnx.data',
                    data: "../published models/best_decoder.onnx.data"
                }
            ]
        });
        console.log("Model loaded successfully");
    } catch (error) {
        console.error("Error loading model:", error);
    }
}

// Initialize model when page loads
initializeModel();

// set canvas scale and position
ctx.translate(canvas.width/2, canvas.height/2);
ctx.scale(minimumaspect, minimumaspect);

plot_latent_space(zoom, panx, pany);

canvas.addEventListener("wheel", scroll)
canvas.addEventListener("mousedown", mousedown)
canvas.addEventListener("mouseup", mouseup)
canvas.addEventListener("mousemove", mousemove)
canvas.addEventListener("click", canvasClick)
canvas.addEventListener("touchstart", touchStart)
canvas.addEventListener("touchmove", touchMove)
canvas.addEventListener("touchend", touchEnd)

window.addEventListener("resize", handleCanvasResize)


// var addButton = document.getElementById('addDatapointButton');
// addButton.addEventListener('click', addDatapoint);

var playButton = document.getElementById('playAudioButton');
playButton.addEventListener('click', playAudio);

var generateButton = document.getElementById('generateHybridButton');
generateButton.addEventListener('click', generateHybrid);

var downloadButton = document.getElementById('downloadAudioButton');
downloadButton.addEventListener('click', downloadAudio);

var volumeSlider = document.getElementById('volumeSlider');
volumeSlider.addEventListener('input', updateVolume);

var autoScaleCheckbox = document.getElementById('autoScaleCheckbox');
autoScaleCheckbox.addEventListener('change', function() {
    autoScaleVolume = this.checked;
});

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

function canvasClick(event) {

    // console.log(zoom, panx, pany)

    console.log(mode)

    if (mode === 'selecting') {
        // Convert mouse position to canvas coordinates
        const rect = canvas.getBoundingClientRect();
        const mouseX = event.clientX - rect.left;
        const mouseY = event.clientY - rect.top;

        // Convert to canvas context coordinates (accounting for transforms)
        const canvasX = (mouseX - canvas.width/2) / (minimumaspect * zoom) - panx;
        const canvasY = (mouseY - canvas.height/2) / (minimumaspect * zoom) - pany;

        // Find closest datapoint
        let closestIndex = -1;
        let closestDistance = Math.max(0.005/zoom, 0.02); // Threshold for clicking on a point
        const radius = Math.max(0.005/zoom, 0.02);

        for (let i = 0; i < embeddingData.length; i++) {
            const point = embeddingData[i].UMAP;
            const dx = canvasX - point[0];
            const dy = canvasY - point[1];
            const distance = Math.sqrt(dx*dx + dy*dy);

            if (distance < closestDistance) {
                closestDistance = distance;
                closestIndex = i;
            }
        }

        if (closestIndex !== -1) {
            addToSelection(closestIndex);
        }
    }
}

function plot_latent_space(zoom, panx, pany){

    ctx.fillStyle = "black";
    ctx.fillRect(-1*aspectRatio, -1*aspectRatio,2*aspectRatio,2*aspectRatio);

    ctx.save();

    ctx.scale(zoom, zoom)
    ctx.translate(panx, pany)
    const fontsize = Math.max(15/(zoom*minimumaspect), 0.05)

    ctx.font = `${fontsize}px Arial`;

    const radius = Math.max(0.005/zoom, 0.01);

    for(var i = 0; i < embeddingData.length; i++){

        const chunkData = embeddingData[i];
        const UMAP = chunkData.UMAP;

        // Highlight selected points
        if (selectedIndices.includes(i)) {
            ctx.beginPath();
            ctx.arc(UMAP[0], UMAP[1], radius * 2, 0, 2 * Math.PI);
            ctx.fillStyle = "rgba(255, 255, 255, 0.8)";
            ctx.fill();
        }

        ctx.beginPath();
        ctx.arc(UMAP[0], UMAP[1], radius, 0, 2 * Math.PI);
        ctx.fillStyle = colors[i];
        ctx.fill();

        if(zoom > 0.3){
            ctx.fillStyle = "rgba(255,255,255,0.5)"; 
            ctx.fillText(chunkData.label, UMAP[0]-fontsize*5, UMAP[1]+radius);
        }
    }

    if(selectedIndices.length > 1){
        for(var i = 0; i < selectedIndices.length; i++){
            const index = selectedIndices[i];
            const chunkData = embeddingData[index];
            const UMAP = chunkData.UMAP;

            ctx.setLineDash([radius*2, radius]); 
            ctx.strokeStyle = "rgba(255, 255, 255, 0.8)";
            ctx.lineWidth = radius;

            // 2. Draw the path as usual
            ctx.beginPath();
            ctx.moveTo(UMAP[0], UMAP[1]);
            ctx.lineTo(comX, comY);

            // 3. Render the stroke
            ctx.stroke();
        }
    }``

    ctx.restore();
}

// UI Functions
function addDatapoint() {
    mode = 'selecting';
    updateModeIndicator();
    console.log("Switched to selecting mode - click on datapoints to add them");
}

function addToSelection(index) {
    if (!selectedIndices.includes(index)) {
        selectedIndices.push(index);
        updateSelectionUI();
        calculateCOM();
        plot_latent_space(zoom, panx, pany);
    }
}

function calculateCOM() {
    if (selectedIndices.length === 0) return;
    let sumX = 0, sumY = 0;
    selectedIndices.forEach(i => {
        sumX += embeddingData[i].UMAP[0];
        sumY += embeddingData[i].UMAP[1];
    }
    );
    comX = sumX / selectedIndices.length;
    comY = sumY / selectedIndices.length;
}

function removeDatapoint(element) {
    const tagElement = element.closest('.datapoint-tag');
    const index = Array.from(document.querySelectorAll('.datapoint-tag')).indexOf(tagElement);
    
    // Find the actual index in selectedIndices
    if (index !== -1 && index < selectedIndices.length) {
        const removedIndex = selectedIndices[index];
        selectedIndices.splice(index, 1);
        updateSelectionUI();
        calculateCOM();
        plot_latent_space(zoom, panx, pany);
    }
}

function updateSelectionUI() {
    const container = document.getElementById('selected-datapoints');
    container.innerHTML = '';

    if (selectedIndices.length === 0) {
        container.innerHTML = '<div class="empty-state">No datapoints selected</div>';
        return;
    }

    selectedIndices.forEach(index => {
        const data = embeddingData[index];
        const color = colors[index];
        
        const tag = document.createElement('div');
        tag.className = 'datapoint-tag';
        tag.style.backgroundColor = color;
        
        const name = document.createElement('span');
        name.className = 'datapoint-tag-name';
        name.textContent = data.label;
        
        const removeBtn = document.createElement('button');
        removeBtn.className = 'datapoint-tag-remove';
        removeBtn.textContent = '×';
        removeBtn.onclick = function() { removeDatapoint(this); };
        
        tag.appendChild(name);
        tag.appendChild(removeBtn);
        container.appendChild(tag);
    });
}

function updateModeIndicator() {
    const indicator = document.getElementById('mode-indicator');
    indicator.textContent = mode.charAt(0).toUpperCase() + mode.slice(1);
    indicator.className = `mode-indicator ${mode}`;
}

function updateVolume() {
    const slider = document.getElementById('volumeSlider');
    const valueDisplay = document.getElementById('volumeValue');
    volume = parseInt(slider.value) / 100;
    valueDisplay.textContent = slider.value;
}

function calculateAutoScaleFactor(audioData) {
    // Find peak amplitude
    let peak = 0;
    for (let i = 0; i < audioData.length; i++) {
        peak = Math.max(peak, Math.abs(audioData[i]));
    }

    // If peak is very quiet (below 0.1), amplify to ~0.8
    if (peak < 0.1) {
        return Math.min(0.8 / peak, 10); // Cap scaling at 10x to avoid extreme amplification
    }

    // If peak is already good, no scaling needed
    return 1.0;
}

async function generateHybrid() {

    console.log("works")
    if (selectedIndices.length === 0) {
        console.log("No datapoints selected");
        return;
    }

    // Average the selected embeddings
    const selectedEmbeddings = selectedIndices.map(i => embeddingData[i].features);
    const averagedEmbedding = new Float32Array(selectedEmbeddings[0].length);
    
    for (let i = 0; i < averagedEmbedding.length; i++) {
        let sum = 0;
        for (let j = 0; j < selectedEmbeddings.length; j++) {
            sum += selectedEmbeddings[j][i];
        }
        averagedEmbedding[i] = sum / selectedEmbeddings.length;
    }

    // Generate spectrogram from latent vector
    await generateSpectrogramFromLatent(averagedEmbedding);

    // Switch back to panning mode
    // mode = 'panning';
    updateModeIndicator();
}

async function generateSpectrogramFromLatent(latentVector) {
    if (!session) {
        console.error("Model not loaded");
        return;
    }

    try {
        const inputTensor = new ort.Tensor('float32', latentVector, [1, 128]);
        const feeds = { embedding: inputTensor };
        const results = await session.run(feeds);
        
        const specOutput = results.spectrogram;
        const specData = specOutput.data;
        const dims = specOutput.dims || [];

        let nMels = 128;
        let nFrames = specData.length / nMels;

        if (dims.length === 4) {
            nMels = dims[2];
            nFrames = dims[3];
        } else if (dims.length === 2) {
            nMels = dims[0];
            nFrames = dims[1];
        }

        const { audio, sr } = melSpecDbToAudio(specData, [1, 1, nMels, nFrames], melPinv, {
            sr: 22050,
            nFft: 2048,
            hopLength: 512,
            nIter: 32,
        });

        currentAudioData = audio;
        currentAudioSampleRate = sr;

        // Render spectrogram on canvas
        renderSpectrogramOnCanvas(specData, nMels, nFrames);

    } catch (error) {
        console.error("Error generating spectrogram:", error);
    }
}

function renderSpectrogramOnCanvas(spectrogramData, nMels, nFrames) {
    const canvas = document.getElementById('spectrogram-canvas');
    const ctx = canvas.getContext('2d');

    // Store for redrawing with progress line
    currentSpectrogramData = spectrogramData;
    currentSpectrogramNMels = nMels;
    currentSpectrogramNFrames = nFrames;

    const cssWidth = canvas.offsetWidth;
    const cssHeight = canvas.offsetHeight;
    const scale = window.devicePixelRatio || 1;

    canvas.width = Math.round(cssWidth * scale);
    canvas.height = Math.round(cssHeight * scale);
    ctx.setTransform(scale, 0, 0, scale, 0, 0);

    ctx.clearRect(0, 0, cssWidth, cssHeight);

    const width = cssWidth;
    const height = cssHeight;

    // Find min and max for normalization
    let min = Infinity, max = -Infinity;
    for (let i = 0; i < spectrogramData.length; i++) {
        min = Math.min(min, spectrogramData[i]);
        max = Math.max(max, spectrogramData[i]);
    }

    const range = max - min || 1;

    // Draw spectrogram with correct row/column mapping
    const pixelWidth = width / nFrames;
    const pixelHeight = height / nMels;

    for (let mel = 0; mel < nMels; mel++) {
        for (let frame = 0; frame < nFrames; frame++) {
            const index = mel * nFrames + frame;
            const normalized = (spectrogramData[index] - min) / range;
            const intensity = Math.floor(normalized * 255);
            ctx.fillStyle = `rgb(${intensity}, ${intensity}, ${intensity})`;
            ctx.fillRect(frame * pixelWidth, (nMels - mel - 1) * pixelHeight, pixelWidth, pixelHeight);
        }
    }
}

function playAudio() {
    if (!currentAudioData) {
        console.log("No audio generated yet");
        return;
    }

    // Calculate auto-scale factor if enabled
    let scaleFactor = 1.0;
    if (autoScaleVolume) {
        scaleFactor = calculateAutoScaleFactor(currentAudioData);
    }

    // Apply volume and auto-scaling to audio data
    const volumedAudio = new Float32Array(currentAudioData.length);
    for (let i = 0; i < currentAudioData.length; i++) {
        volumedAudio[i] = currentAudioData[i] * volume * scaleFactor;
    }

    playbackDuration = currentAudioData.length / currentAudioSampleRate;
    playbackStartTime = Date.now();
    isPlaying = true;

    // Start the playback progress animation
    updatePlaybackProgress();

    playFloatPCM(volumedAudio, currentAudioSampleRate);
}

function updatePlaybackProgress() {
    if (!isPlaying) {
        return;
    }

    const elapsed = (Date.now() - playbackStartTime) / 1000;
    const progress = Math.min(elapsed / playbackDuration, 1);

    drawProgressLine(progress);

    if (progress < 1) {
        playbackAnimationId = requestAnimationFrame(updatePlaybackProgress);
    } else {
        isPlaying = false;
        // Clear the progress line when done
        drawProgressLine(0);
    }
}

function drawProgressLine(progress) {
    if (!currentSpectrogramData) return;

    const canvas = document.getElementById('spectrogram-canvas');
    const ctx = canvas.getContext('2d');

    const cssWidth = canvas.offsetWidth;
    const cssHeight = canvas.offsetHeight;
    const scale = window.devicePixelRatio || 1;

    // Redraw spectrogram
    const width = cssWidth;
    const height = cssHeight;

    // Find min and max for normalization
    let min = Infinity, max = -Infinity;
    for (let i = 0; i < currentSpectrogramData.length; i++) {
        min = Math.min(min, currentSpectrogramData[i]);
        max = Math.max(max, currentSpectrogramData[i]);
    }

    const range = max - min || 1;

    // Redraw spectrogram
    const pixelWidth = width / currentSpectrogramNFrames;
    const pixelHeight = height / currentSpectrogramNMels;

    for (let mel = 0; mel < currentSpectrogramNMels; mel++) {
        for (let frame = 0; frame < currentSpectrogramNFrames; frame++) {
            const index = mel * currentSpectrogramNFrames + frame;
            const normalized = (currentSpectrogramData[index] - min) / range;
            const intensity = Math.floor(normalized * 255);
            ctx.fillStyle = `rgb(${intensity}, ${intensity}, ${intensity})`;
            ctx.fillRect(frame * pixelWidth, (currentSpectrogramNMels - mel - 1) * pixelHeight, pixelWidth, pixelHeight);
        }
    }

    // Draw progress line
    const xPos = progress * width;
    ctx.strokeStyle = 'rgba(255, 100, 100, 0.9)';
    ctx.lineWidth = 2 / scale;
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.moveTo(xPos, 0);
    ctx.lineTo(xPos, height);
    ctx.stroke();
}

function downloadAudio() {
    if (!currentAudioData || !currentAudioSampleRate) {
        console.log("No audio generated yet");
        return;
    }

    // Calculate auto-scale factor if enabled
    let scaleFactor = 1.0;
    if (autoScaleVolume) {
        scaleFactor = calculateAutoScaleFactor(currentAudioData);
    }

    const scaledAudioData = new Float32Array(currentAudioData.length);
    for (let i = 0; i < currentAudioData.length; i++) {
        scaledAudioData[i] = currentAudioData[i] * volume * scaleFactor;
    }

    // Convert float PCM to WAV
    const numberOfChannels = 1;
    const sampleRate = currentAudioSampleRate;
    const format = 1; // PCM
    const bitDepth = 16;

    const bytesPerSample = bitDepth / 8;
    const blockAlign = numberOfChannels * bytesPerSample;

    // Create WAV header
    const arrayBuffer = new ArrayBuffer(44 + scaledAudioData.length * bytesPerSample);
    const view = new DataView(arrayBuffer);

    // WAV header
    const writeString = (offset, string) => {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    };

    writeString(0, 'RIFF');
    view.setUint32(4, 36 + scaledAudioData.length * bytesPerSample, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, format, true);
    view.setUint16(22, numberOfChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * blockAlign, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bitDepth, true);
    writeString(36, 'data');
    view.setUint32(40, scaledAudioData.length * bytesPerSample, true);

    // Convert float to 16-bit PCM
    let offset = 44;
    for (let i = 0; i < scaledAudioData.length; i++) {
        const sample = Math.max(-1, Math.min(1, scaledAudioData[i]));
        view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
        offset += 2;
    }

    // Create blob and download
    const blob = new Blob([arrayBuffer], { type: 'audio/wav' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'bird_song.wav';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// Touch event handlers for mobile
function getTouchDistance(touches) {
    if (touches.length < 2) return 0;
    const dx = touches[0].clientX - touches[1].clientX;
    const dy = touches[0].clientY - touches[1].clientY;
    return Math.sqrt(dx * dx + dy * dy);
}

function touchStart(event) {
    if (event.touches.length === 2) {
        lastTouchDistance = getTouchDistance(event.touches);
        isTouching = true;
    } else if (event.touches.length === 1) {
        panning = true;
        mouseposx = event.touches[0].clientX;
        mouseposy = event.touches[0].clientY;
        isTouching = true;
    }
}

function touchMove(event) {
    if (event.touches.length === 2) {
        // Pinch to zoom
        touchDistance = getTouchDistance(event.touches);
        const zoomFactor = touchDistance / lastTouchDistance;
        zoom *= zoomFactor;
        lastTouchDistance = touchDistance;
        plot_latent_space(zoom, panx, pany);
    } else if (event.touches.length === 1 && panning) {
        // Drag to pan
        const dx = event.touches[0].clientX - mouseposx;
        const dy = event.touches[0].clientY - mouseposy;
        panx += (dx / zoom) / minimumaspect;
        pany += (dy / zoom) / minimumaspect;
        plot_latent_space(zoom, panx, pany);
        
        mouseposx = event.touches[0].clientX;
        mouseposy = event.touches[0].clientY;
    }
}

function touchEnd(event) {
    panning = false;
    isTouching = false;
}

// Mobile zoom controls
function mobileZoomIn() {
    zoom *= 1.2;
    plot_latent_space(zoom, panx, pany);
}

function mobileZoomOut() {
    zoom /= 1.2;
    plot_latent_space(zoom, panx, pany);
}

// Create mobile controls
function createMobileControls() {
    const controlsDiv = document.createElement('div');
    controlsDiv.id = 'mobile-controls';
    controlsDiv.style.cssText = `
        position: fixed;
        bottom: 20px;
        left: 20px;
        display: flex;
        flex-direction: column;
        gap: 8px;
        z-index: 100;
        display: none;
    `;
    
    // Check if mobile
    if (window.innerWidth <= 768) {
        controlsDiv.style.display = 'flex';
    }
    
    const zoomInBtn = document.createElement('button');
    zoomInBtn.textContent = '🔍+';
    zoomInBtn.style.cssText = `
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background-color: #4f46e5;
        color: white;
        border: none;
        cursor: pointer;
        font-size: 18px;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
        transition: all 0.2s;
    `;
    zoomInBtn.onclick = mobileZoomIn;
    zoomInBtn.onmouseover = () => zoomInBtn.style.backgroundColor = '#4338ca';
    zoomInBtn.onmouseout = () => zoomInBtn.style.backgroundColor = '#4f46e5';
    
    const zoomOutBtn = document.createElement('button');
    zoomOutBtn.textContent = '🔍−';
    zoomOutBtn.style.cssText = `
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background-color: #4f46e5;
        color: white;
        border: none;
        cursor: pointer;
        font-size: 18px;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
        transition: all 0.2s;
    `;
    zoomOutBtn.onclick = mobileZoomOut;
    zoomOutBtn.onmouseover = () => zoomOutBtn.style.backgroundColor = '#4338ca';
    zoomOutBtn.onmouseout = () => zoomOutBtn.style.backgroundColor = '#4f46e5';
    
    controlsDiv.appendChild(zoomInBtn);
    controlsDiv.appendChild(zoomOutBtn);
    document.body.appendChild(controlsDiv);
}

// Initialize mobile controls
createMobileControls();

// Show/hide mobile controls on resize
window.addEventListener('resize', () => {
    const controlsDiv = document.getElementById('mobile-controls');
    if (controlsDiv) {
        if (window.innerWidth <= 768) {
            controlsDiv.style.display = 'flex';
        } else {
            controlsDiv.style.display = 'none';
        }
    }
});
