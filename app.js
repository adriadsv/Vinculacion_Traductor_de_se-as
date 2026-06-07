const video = document.querySelector("#webcam");
const canvas = document.querySelector("#overlay");
const ctx = canvas.getContext("2d");
const cameraButton = document.querySelector("#cameraButton");
const flipButton = document.querySelector("#flipButton");
const modeButton = document.querySelector("#modeButton");
const clearPhraseButton = document.querySelector("#clearPhraseButton");
const adminToggleButton = document.querySelector("#adminToggleButton");
const adminLoginPanel = document.querySelector("#adminLoginPanel");
const adminUserInput = document.querySelector("#adminUserInput");
const adminPasswordInput = document.querySelector("#adminPasswordInput");
const adminSubmitButton = document.querySelector("#adminSubmitButton");
const adminStatusText = document.querySelector("#adminStatusText");
const captureButton = document.querySelector("#captureButton");
const captureMotionButton = document.querySelector("#captureMotionButton");
const saveButton = document.querySelector("#saveButton");
const captureLetterButton = document.querySelector("#captureLetterButton");
const captureLetterMotionButton = document.querySelector("#captureLetterMotionButton");
const saveLetterButton = document.querySelector("#saveLetterButton");
const signNameInput = document.querySelector("#signNameInput");
const signScopeText = document.querySelector("#signScopeText");
const signSearchInput = document.querySelector("#signSearchInput");
const letterSelect = document.querySelector("#letterSelect");
const letterScopeText = document.querySelector("#letterScopeText");
const letterSearchInput = document.querySelector("#letterSearchInput");
const modelStatus = document.querySelector("#modelStatus");
const cameraEmpty = document.querySelector("#cameraEmpty");
const translationText = document.querySelector("#translationText");
const confidenceText = document.querySelector("#confidenceText");
const phraseText = document.querySelector("#phraseText");
const fpsLabel = document.querySelector("#fpsLabel");
const sampleCounter = document.querySelector("#sampleCounter");
const pendingSampleList = document.querySelector("#pendingSampleList");
const signList = document.querySelector("#signList");
const letterStatus = document.querySelector("#letterStatus");
const letterOutput = document.querySelector("#letterOutput");
const letterSampleCounter = document.querySelector("#letterSampleCounter");
const pendingLetterSampleList = document.querySelector("#pendingLetterSampleList");
const letterList = document.querySelector("#letterList");

const STORAGE_KEY = "vinculacion.lsec.samples.v1";
const LETTER_STORAGE_KEY = "vinculacion.lsec.letters.v1";
const ADMIN_TOKEN_KEY = "vinculacion.lsec.adminToken.v1";
const HOLD_TO_APPEND_MS = 1100;
const CUSTOM_MATCH_THRESHOLD = 0.19;
const LETTER_MATCH_THRESHOLD = 0.18;
const FACE_EVERY_FRAMES = 2;
const MOTION_CAPTURE_INTERVAL_MS = 90;
const MOTION_WINDOW_MS = 2200;
const MIN_MOTION_FRAMES = 4;
const MOTION_RESAMPLE_FRAMES = 12;
const SEARCH_ENABLE_THRESHOLD = 5;
const HAND_VECTOR_SIZE = 63;
const FACE_LANDMARK_INDICES = [
  10, 33, 61, 78, 133, 152, 159, 145, 263, 291, 308, 362, 386, 374
];
const FACE_BLENDSHAPE_NAMES = [
  "browDownLeft",
  "browDownRight",
  "browInnerUp",
  "eyeBlinkLeft",
  "eyeBlinkRight",
  "jawOpen",
  "mouthFrownLeft",
  "mouthFrownRight",
  "mouthPucker",
  "mouthSmileLeft",
  "mouthSmileRight"
];
const ALPHABET = [
  "A", "B", "C", "CH", "D", "E", "F", "G", "H", "I", "J", "K",
  "L", "LL", "LLL", "M", "N", "\u00d1", "O", "P", "Q", "R", "RR",
  "S", "T", "U", "V", "W", "X", "Y", "Z"
];
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [0, 17], [17, 18], [18, 19], [19, 20]
];

let gestureRecognizer;
let faceLandmarker;
let drawingUtils;
let DrawingUtils;
let FaceLandmarker;
let FilesetResolver;
let GestureRecognizer;
let stream;
let facingMode = "user";
let running = false;
let lastVideoTime = -1;
let lastFrameAt = performance.now();
let frameIndex = 0;
let lastFaceResult = null;
let phrase = [];
let activePrediction = { label: "Esperando senia...", score: 0 };
let holdLabel = "";
let holdStart = 0;
let pendingSamples = [];
let pendingLetterSamples = [];
let customSigns = loadSigns();
let customLetters = loadLetters();
let globalSigns = [];
let globalLetters = [];
let adminToken = sessionStorage.getItem(ADMIN_TOKEN_KEY) ?? "";
let mode = "words";
let motionRecording = null;
let recentMotionFrames = [];

const baseTranslations = {
  Open_Palm: "hola",
  Closed_Fist: "si / afirmacion",
  Thumb_Up: "bien",
  Pointing_Up: "atencion",
  Victory: "dos / paz",
  Thumb_Down: "no / negativo",
  ILoveYou: "te quiero"
};

init();

async function init() {
  setControlsEnabled(false);
  updateAdminUi();
  renderPendingSamples("sign");
  renderPendingSamples("letter");
  renderSignList();
  renderLetterSelect();
  renderLetterList();
  await loadGlobalData();

  try {
    const mediaPipe = await import("./node_modules/@mediapipe/tasks-vision/vision_bundle.mjs");
    DrawingUtils = mediaPipe.DrawingUtils;
    FaceLandmarker = mediaPipe.FaceLandmarker;
    FilesetResolver = mediaPipe.FilesetResolver;
    GestureRecognizer = mediaPipe.GestureRecognizer;

    const vision = await FilesetResolver.forVisionTasks(
      "./node_modules/@mediapipe/tasks-vision/wasm"
    );

    gestureRecognizer = await GestureRecognizer.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          "./models/gesture_recognizer.task",
        delegate: "GPU"
      },
      runningMode: "VIDEO",
      numHands: 2
    });

    faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          "./models/face_landmarker.task",
        delegate: "GPU"
      },
      runningMode: "VIDEO",
      numFaces: 1,
      outputFaceBlendshapes: true
    });

    drawingUtils = new DrawingUtils(ctx);
    modelStatus.textContent = "Modelos listos";
    setControlsEnabled(true);
  } catch (error) {
    console.error(error);
    modelStatus.textContent = "No se pudieron cargar los modelos";
    translationText.textContent = "No cargo MediaPipe";
    confidenceText.textContent = "Revisa internet, recarga la pagina o mira la consola del navegador.";
  }
}

function setControlsEnabled(enabled) {
  cameraButton.disabled = !enabled;
  flipButton.disabled = !enabled;
  captureButton.disabled = !enabled;
  captureMotionButton.disabled = !enabled;
  saveButton.disabled = !enabled;
  captureLetterButton.disabled = !enabled;
  captureLetterMotionButton.disabled = !enabled;
  saveLetterButton.disabled = !enabled;
}

cameraButton.addEventListener("click", async () => {
  if (running) {
    stopCamera();
    return;
  }
  await startCamera();
});

flipButton.addEventListener("click", async () => {
  facingMode = facingMode === "user" ? "environment" : "user";
  if (running) {
    stopCamera();
    await startCamera();
  }
});

clearPhraseButton.addEventListener("click", () => {
  phrase = [];
  phraseText.textContent = "";
});

signSearchInput.addEventListener("input", renderSignList);
letterSearchInput.addEventListener("input", renderLetterList);

adminToggleButton.addEventListener("click", () => {
  if (adminToken) {
    adminToken = "";
    sessionStorage.removeItem(ADMIN_TOKEN_KEY);
    updateAdminUi();
    return;
  }
  adminLoginPanel.classList.toggle("hidden");
});

adminSubmitButton.addEventListener("click", async () => {
  await loginAdmin();
});

adminPasswordInput.addEventListener("keydown", async (event) => {
  if (event.key === "Enter") await loginAdmin();
});

modeButton.addEventListener("click", () => {
  mode = mode === "words" ? "letters" : "words";
  modeButton.textContent = mode === "words" ? "Modo: palabras" : "Modo: abecedario";
  phrase = [];
  phraseText.textContent = "";
});

captureButton.addEventListener("click", () => {
  const vector = currentVector();
  if (!vector) {
    sampleCounter.textContent = "No hay mano ni rostro detectado para capturar";
    return;
  }
  pendingSamples.push({ kind: "pose", vector });
  sampleCounter.textContent = `${pendingSamples.length} muestras listas`;
  renderPendingSamples("sign");
});

captureMotionButton.addEventListener("click", () => {
  toggleMotionCapture("sign");
});

captureLetterButton.addEventListener("click", () => {
  const vector = currentVector();
  if (!vector) {
    letterSampleCounter.textContent = "No hay mano ni rostro detectado para capturar";
    return;
  }
  pendingLetterSamples.push({ kind: "pose", vector });
  letterSampleCounter.textContent = `${pendingLetterSamples.length} muestras de letra`;
  renderPendingSamples("letter");
});

captureLetterMotionButton.addEventListener("click", () => {
  toggleMotionCapture("letter");
});

saveLetterButton.addEventListener("click", async () => {
  const letter = letterSelect.value;
  if (!letter || pendingLetterSamples.length === 0) {
    letterSampleCounter.textContent = "Elige una letra y captura al menos 1 muestra";
    return;
  }
  const item = { label: letter, samples: [...pendingLetterSamples] };
  if (adminToken) {
    const ok = await saveGlobalItem("letters", item);
    if (!ok) return;
  } else {
    const existing = customLetters.find((entry) => entry.label === letter);
    if (existing) {
      existing.samples.push(...pendingLetterSamples);
    } else {
      customLetters.push(item);
    }
    saveLetters(customLetters);
  }
  pendingLetterSamples = [];
  letterSampleCounter.textContent = "0 muestras de letra";
  renderPendingSamples("letter");
  renderLetterList();
});

saveButton.addEventListener("click", async () => {
  const name = signNameInput.value.trim().toLowerCase();
  if (!name || pendingSamples.length === 0) {
    sampleCounter.textContent = "Escribe un nombre y captura al menos 1 muestra";
    return;
  }
  const item = { label: name, samples: [...pendingSamples] };
  if (adminToken) {
    const ok = await saveGlobalItem("signs", item);
    if (!ok) return;
  } else {
    const existing = customSigns.find((sign) => sign.label === name);
    if (existing) {
      existing.samples.push(...pendingSamples);
    } else {
      customSigns.push(item);
    }
    saveSigns(customSigns);
  }
  pendingSamples = [];
  signNameInput.value = "";
  sampleCounter.textContent = "0 muestras listas";
  renderPendingSamples("sign");
  renderSignList();
});

async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode,
        width: { ideal: 640 },
        height: { ideal: 480 },
        frameRate: { ideal: 30, max: 30 }
      },
      audio: false
    });
    video.srcObject = stream;
    await video.play();
    running = true;
    cameraButton.textContent = "Detener camara";
    cameraEmpty.classList.add("hidden");
    requestAnimationFrame(predictFrame);
  } catch (error) {
    console.error(error);
    modelStatus.textContent = "Permiso de camara denegado o no disponible";
  }
}

function stopCamera() {
  running = false;
  stopMotionCapture(true);
  stream?.getTracks().forEach((track) => track.stop());
  stream = null;
  video.srcObject = null;
  cameraButton.textContent = "Iniciar camara";
  cameraEmpty.classList.remove("hidden");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function predictFrame(now) {
  if (!running) return;

  resizeCanvas();
  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const gestureResult = gestureRecognizer.recognizeForVideo(video, now);
    frameIndex += 1;
    const faceResult = frameIndex % FACE_EVERY_FRAMES === 0
      ? faceLandmarker.detectForVideo(video, now)
      : lastFaceResult;
    if (frameIndex % FACE_EVERY_FRAMES === 0) lastFaceResult = faceResult;
    window.lastHandLandmarks = gestureResult.landmarks ?? window.lastHandLandmarks;
    window.lastFaceResult = faceResult ?? window.lastFaceResult;
    const fullVector = currentVector(gestureResult, faceResult);
    trackMotionFrame(fullVector, now);
    renderDetections(gestureResult, faceResult);
    updatePrediction(gestureResult, faceResult, now);
    updateFps(now);
  }

  requestAnimationFrame(predictFrame);
}

function resizeCanvas() {
  const width = video.videoWidth || canvas.clientWidth;
  const height = video.videoHeight || canvas.clientHeight;
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
}

function renderDetections(gestureResult, faceResult) {
  ctx.save();
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.scale(-1, 1);
  ctx.translate(-canvas.width, 0);

  for (const landmarks of gestureResult.landmarks ?? []) {
    drawHandLines(landmarks);
    drawingUtils.drawLandmarks(landmarks, {
      color: "#f2c14e",
      lineWidth: 2,
      radius: 5
    });
  }

  for (const landmarks of faceResult?.faceLandmarks ?? []) {
    drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_TESSELATION, {
      color: "rgba(244, 242, 237, 0.22)",
      lineWidth: 1
    });
    drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_FACE_OVAL, {
      color: "#f2c14e",
      lineWidth: 2
    });
  }

  ctx.restore();
}

function drawHandLines(landmarks) {
  ctx.save();
  ctx.strokeStyle = "#2ec4b6";
  ctx.lineWidth = Math.max(3, canvas.width * 0.004);
  ctx.lineCap = "round";
  ctx.lineJoin = "round";

  for (const [start, end] of HAND_CONNECTIONS) {
    const a = landmarks[start];
    const b = landmarks[end];
    ctx.beginPath();
    ctx.moveTo(a.x * canvas.width, a.y * canvas.height);
    ctx.lineTo(b.x * canvas.width, b.y * canvas.height);
    ctx.stroke();
  }

  ctx.restore();
}

function updatePrediction(gestureResult, faceResult, now) {
  const fullVector = currentVector(gestureResult, faceResult);
  const letter = classifyLetter(gestureResult, fullVector);
  letterOutput.textContent = letter?.label ?? "--";
  letterStatus.textContent = letter ? `${Math.round(letter.score * 100)}%` : "LSEC";

  if (mode === "letters") {
    activePrediction = letter
      ? { label: letter.label, score: letter.score }
      : { label: "Esperando letra...", score: 0 };
    translationText.textContent = activePrediction.label;
    confidenceText.textContent = `Confianza: ${Math.round(activePrediction.score * 100)}%`;
    maybeAppendPhrase(activePrediction.label, now);
    return;
  }

  const custom = classifyCustom(fullVector, gestureResult);
  const canned = classifyBaseGesture(gestureResult);
  const expression = classifyExpression(faceResult);

  if (custom && custom.score > CUSTOM_MATCH_THRESHOLD) {
    activePrediction = custom;
  } else if (canned) {
    activePrediction = canned;
  } else if (expression) {
    activePrediction = expression;
  } else {
    activePrediction = { label: "Esperando senia...", score: 0 };
  }

  translationText.textContent = activePrediction.label;
  confidenceText.textContent = `Confianza: ${Math.round(activePrediction.score * 100)}%`;
  maybeAppendPhrase(activePrediction.label, now);
}

function classifyBaseGesture(gestureResult) {
  const category = gestureResult.gestures?.[0]?.[0];
  if (!category || category.categoryName === "None") return null;
  return {
    label: baseTranslations[category.categoryName] ?? category.categoryName,
    score: category.score ?? 0
  };
}

function classifyExpression(faceResult) {
  const blendshapes = faceResult?.faceBlendshapes?.[0]?.categories ?? [];
  const smile = blendshapes.find((item) => item.categoryName === "mouthSmileLeft")?.score ?? 0;
  const brow = blendshapes.find((item) => item.categoryName === "browInnerUp")?.score ?? 0;
  if (smile > 0.55) return { label: "expresion: sonrisa", score: smile };
  if (brow > 0.55) return { label: "expresion: pregunta", score: brow };
  return null;
}

function classifyLetter(gestureResult, fullVector = null) {
  const trained = classifyCustomLetter(fullVector, gestureResult);
  if (trained && trained.score > LETTER_MATCH_THRESHOLD) return trained;

  const landmarks = gestureResult?.landmarks?.[0];
  if (!landmarks) return null;

  return classifyLetterByShape(gestureResult.landmarks);
}

function classifyCustomLetter(vector, gestureResult = null) {
  const letters = [...globalLetters, ...customLetters];
  if (!vector || letters.length === 0) return null;
  let best = { label: "", distance: Infinity };

  for (const letter of letters) {
    for (const sample of letter.samples) {
      const distance = sampleDistance(vector, sample, gestureResult);
      if (distance < best.distance) best = { label: letter.label, distance };
    }
  }

  const score = Math.max(0, 1 - best.distance / 2.8);
  return { label: best.label, score };
}

function classifyLetterByShape(hands) {
  const landmarks = hands[0];
  const secondHand = hands[1];
  const fingers = fingerStates(landmarks);
  const thumbUp = fingers.thumb;
  const indexUp = fingers.index;
  const middleUp = fingers.middle;
  const ringUp = fingers.ring;
  const pinkyUp = fingers.pinky;
  const extendedCount = [indexUp, middleUp, ringUp, pinkyUp].filter(Boolean).length;
  const thumbIndexDistance = distance2d(landmarks[4], landmarks[8]);
  const thumbMiddleDistance = distance2d(landmarks[4], landmarks[12]);
  const indexMiddleDistance = distance2d(landmarks[8], landmarks[12]);

  let label = null;
  let score = 0.55;

  if (secondHand && isLShape(landmarks) && isLShape(secondHand)) label = "LLL";
  else if (!thumbUp && extendedCount === 0) label = "A";
  else if (!thumbUp && extendedCount === 4) label = "B";
  else if (looksLikeC(landmarks)) label = "C";
  else if (extendedCount === 4 && thumbIndexDistance < 0.09) label = "F";
  else if (indexUp && middleUp && !ringUp && !pinkyUp && indexMiddleDistance > 0.06) label = "V";
  else if (indexUp && middleUp && ringUp && !pinkyUp) label = "W";
  else if (indexUp && !middleUp && !ringUp && !pinkyUp) label = "D";
  else if (!indexUp && !middleUp && !ringUp && pinkyUp) label = "I";
  else if (thumbUp && !indexUp && !middleUp && !ringUp && pinkyUp) label = "Y";
  else if (isLShape(landmarks)) label = "L";
  else if (!thumbUp && !indexUp && middleUp && !ringUp && !pinkyUp) label = "E";
  else if (!thumbUp && indexUp && middleUp && ringUp && pinkyUp) label = "B";
  else if (extendedCount === 0 && thumbIndexDistance < 0.08) label = "O";
  else if (indexUp && middleUp && !ringUp && !pinkyUp && thumbMiddleDistance < 0.12) label = "K";
  else if (thumbUp && indexUp && middleUp && !ringUp && !pinkyUp) label = "CH";

  if (!label) return null;
  return { label, score };
}

function isLShape(landmarks) {
  const fingers = fingerStates(landmarks);
  return fingers.thumb && fingers.index && !fingers.middle && !fingers.ring && !fingers.pinky;
}

function looksLikeC(landmarks) {
  const fingers = fingerStates(landmarks);
  const thumbIndexDistance = distance2d(landmarks[4], landmarks[8]);
  const thumbPinkyDistance = distance2d(landmarks[4], landmarks[20]);
  return !fingers.index && !fingers.middle && !fingers.ring && !fingers.pinky
    && thumbIndexDistance > 0.09
    && thumbPinkyDistance > 0.14;
}

function fingerStates(landmarks) {
  const handedness = landmarks[17].x < landmarks[5].x ? "right" : "left";
  return {
    thumb: handedness === "right" ? landmarks[4].x < landmarks[3].x : landmarks[4].x > landmarks[3].x,
    index: landmarks[8].y < landmarks[6].y,
    middle: landmarks[12].y < landmarks[10].y,
    ring: landmarks[16].y < landmarks[14].y,
    pinky: landmarks[20].y < landmarks[18].y
  };
}

function distance2d(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function currentHandVector(gestureResult = null) {
  const landmarks = gestureResult?.landmarks?.[0] ?? window.lastHandLandmarks?.[0];
  if (!landmarks) return null;
  window.lastHandLandmarks = gestureResult?.landmarks ?? window.lastHandLandmarks;
  return normalizeHand(landmarks);
}

function normalizeHand(landmarks) {
  const wrist = landmarks[0];
  const middle = landmarks[9];
  const scale = Math.hypot(middle.x - wrist.x, middle.y - wrist.y, middle.z - wrist.z) || 1;
  return landmarks.flatMap((point) => [
    (point.x - wrist.x) / scale,
    (point.y - wrist.y) / scale,
    (point.z - wrist.z) / scale
  ]);
}

function currentVector(gestureResult = null, faceResult = null) {
  const hands = gestureResult?.landmarks ?? window.lastHandLandmarks ?? [];
  const face = faceResult ?? window.lastFaceResult ?? null;
  if (gestureResult?.landmarks) window.lastHandLandmarks = gestureResult.landmarks;
  if (faceResult) window.lastFaceResult = faceResult;

  const sortedHands = [...hands].sort((a, b) => (a[0]?.x ?? 0) - (b[0]?.x ?? 0));
  const firstHand = normalizeOptionalHand(sortedHands[0]);
  const secondHand = normalizeOptionalHand(sortedHands[1]);
  const faceVector = normalizeFace(face);

  if (!firstHand.present && !secondHand.present && !faceVector.present) return null;
  return [
    Number(firstHand.present),
    ...firstHand.values,
    Number(secondHand.present),
    ...secondHand.values,
    Number(faceVector.present),
    ...faceVector.values
  ];
}

function toggleMotionCapture(type) {
  if (motionRecording?.type === type) {
    stopMotionCapture(false);
    return;
  }
  if (motionRecording) stopMotionCapture(true);
  if (!running) {
    const target = type === "letter" ? letterSampleCounter : sampleCounter;
    target.textContent = "Inicia la camara antes de grabar movimiento";
    return;
  }

  motionRecording = {
    type,
    startedAt: performance.now(),
    lastFrameAt: 0,
    frames: []
  };
  const button = type === "letter" ? captureLetterMotionButton : captureMotionButton;
  button.classList.add("recording");
  button.textContent = "Detener movimiento";
  updateMotionStatus(performance.now());
}

function stopMotionCapture(cancelled) {
  if (!motionRecording) return;
  const recording = motionRecording;
  motionRecording = null;

  const button = recording.type === "letter" ? captureLetterMotionButton : captureMotionButton;
  const counter = recording.type === "letter" ? letterSampleCounter : sampleCounter;
  const pending = recording.type === "letter" ? pendingLetterSamples : pendingSamples;

  button.classList.remove("recording");
  button.textContent = "Captura con movimiento";

  if (cancelled) {
    counter.textContent = recording.type === "letter"
      ? `${pendingLetterSamples.length} muestras de letra`
      : `${pendingSamples.length} muestras listas`;
    return;
  }

  if (recording.frames.length < MIN_MOTION_FRAMES) {
    counter.textContent = "Movimiento muy corto o sin tracking suficiente";
    return;
  }

  pending.push({
    kind: "motion",
    duration: recording.frames.at(-1).time - recording.frames[0].time,
    frames: recording.frames.map((frame) => frame.vector)
  });
  counter.textContent = recording.type === "letter"
    ? `${pending.length} muestras de letra`
    : `${pending.length} muestras listas`;
  renderPendingSamples(recording.type === "letter" ? "letter" : "sign");
}

function trackMotionFrame(vector, now) {
  if (vector) {
    recentMotionFrames.push({ time: now, vector });
    recentMotionFrames = recentMotionFrames.filter((frame) => now - frame.time <= MOTION_WINDOW_MS);
  }

  if (!motionRecording) return;
  updateMotionStatus(now);
  if (!vector || now - motionRecording.lastFrameAt < MOTION_CAPTURE_INTERVAL_MS) return;
  motionRecording.frames.push({ time: now - motionRecording.startedAt, vector });
  motionRecording.lastFrameAt = now;
}

function updateMotionStatus(now) {
  if (!motionRecording) return;
  const seconds = Math.max(0, (now - motionRecording.startedAt) / 1000);
  const counter = motionRecording.type === "letter" ? letterSampleCounter : sampleCounter;
  counter.textContent = `Grabando movimiento: ${seconds.toFixed(1)} s`;
}

function normalizeOptionalHand(landmarks) {
  if (!landmarks) {
    return { present: false, values: Array(HAND_VECTOR_SIZE).fill(0) };
  }
  return { present: true, values: normalizeHand(landmarks) };
}

function normalizeFace(faceResult) {
  const landmarks = faceResult?.faceLandmarks?.[0];
  const blendshapes = faceResult?.faceBlendshapes?.[0]?.categories ?? [];
  const blendshapeMap = new Map(blendshapes.map((item) => [item.categoryName, item.score ?? 0]));
  const emptyLandmarks = Array(FACE_LANDMARK_INDICES.length * 3).fill(0);
  const emptyBlendshapes = Array(FACE_BLENDSHAPE_NAMES.length).fill(0);

  if (!landmarks) {
    return { present: false, values: [...emptyLandmarks, ...emptyBlendshapes] };
  }

  const center = landmarks[1] ?? landmarks[0];
  const chin = landmarks[152] ?? center;
  const scale = Math.hypot(chin.x - center.x, chin.y - center.y, chin.z - center.z) || 1;
  const landmarkValues = FACE_LANDMARK_INDICES.flatMap((index) => {
    const point = landmarks[index] ?? center;
    return [
      (point.x - center.x) / scale,
      (point.y - center.y) / scale,
      (point.z - center.z) / scale
    ];
  });
  const blendshapeValues = FACE_BLENDSHAPE_NAMES.map((name) => blendshapeMap.get(name) ?? 0);

  return { present: true, values: [...landmarkValues, ...blendshapeValues] };
}

function classifyCustom(vector, gestureResult = null) {
  const signs = [...globalSigns, ...customSigns];
  if (!vector || signs.length === 0) return null;
  let best = { label: "", distance: Infinity };
  const legacyHandVector = currentHandVector(gestureResult);

  for (const sign of signs) {
    for (const sample of sign.samples) {
      const distance = sampleDistance(vector, sample, gestureResult, legacyHandVector);
      if (distance < best.distance) best = { label: sign.label, distance };
    }
  }

  const score = Math.max(0, 1 - best.distance / 2.8);
  return { label: best.label, score };
}

function sampleDistance(vector, sample, gestureResult = null, legacyHandVector = null) {
  if (sample?.kind === "motion" && Array.isArray(sample.frames)) {
    return motionDistance(currentMotionSequence(), sample.frames);
  }

  const sampleVector = sample?.kind === "pose" && Array.isArray(sample.vector)
    ? sample.vector
    : sample;
  const handVector = legacyHandVector ?? currentHandVector(gestureResult);
  const candidateVector = Array.isArray(sampleVector) && sampleVector.length === HAND_VECTOR_SIZE && handVector
    ? handVector
    : vector;

  if (!candidateVector || !Array.isArray(sampleVector)) return Infinity;
  return euclideanDistance(candidateVector, sampleVector);
}

function currentMotionSequence() {
  return recentMotionFrames.map((frame) => frame.vector);
}

function motionDistance(currentFrames, sampleFrames) {
  if (!currentFrames || !sampleFrames || currentFrames.length < MIN_MOTION_FRAMES) return Infinity;
  const current = resampleFrames(currentFrames, MOTION_RESAMPLE_FRAMES);
  const sample = resampleFrames(sampleFrames, MOTION_RESAMPLE_FRAMES);
  let total = 0;

  for (let index = 0; index < MOTION_RESAMPLE_FRAMES; index += 1) {
    total += euclideanDistance(current[index], sample[index]);
  }
  return total / MOTION_RESAMPLE_FRAMES;
}

function resampleFrames(frames, targetCount) {
  if (frames.length === targetCount) return frames;
  if (frames.length === 1) return Array(targetCount).fill(frames[0]);

  const result = [];
  for (let index = 0; index < targetCount; index += 1) {
    const sourceIndex = Math.round((index / (targetCount - 1)) * (frames.length - 1));
    result.push(frames[sourceIndex]);
  }
  return result;
}

function euclideanDistance(a, b) {
  const length = Math.min(a.length, b.length);
  let total = 0;
  for (let index = 0; index < length; index += 1) {
    const diff = a[index] - b[index];
    total += diff * diff;
  }
  const lengthPenalty = Math.abs(a.length - b.length) / Math.max(a.length, b.length);
  return Math.sqrt(total / length) + lengthPenalty;
}

function maybeAppendPhrase(label, now) {
  if (label === "Esperando senia..." || label === "Esperando letra..." || label.startsWith("expresion:")) {
    holdLabel = "";
    holdStart = 0;
    return;
  }

  if (label !== holdLabel) {
    holdLabel = label;
    holdStart = now;
    return;
  }

  if (now - holdStart > HOLD_TO_APPEND_MS && phrase.at(-1) !== label) {
    phrase.push(label);
    phraseText.textContent = phrase.join(" ");
    holdStart = now + 999999;
  }
}

function updateFps(now) {
  const fps = 1000 / Math.max(1, now - lastFrameAt);
  lastFrameAt = now;
  fpsLabel.textContent = `${Math.round(fps)} fps`;
}

async function loadGlobalData() {
  try {
    const response = await fetch("/api/global-data");
    if (!response.ok) throw new Error("No se pudo cargar el dataset global");
    const data = await response.json();
    globalSigns = Array.isArray(data.signs) ? data.signs : [];
    globalLetters = Array.isArray(data.letters) ? data.letters : [];
    renderSignList();
    renderLetterList();
  } catch (error) {
    console.warn(error);
    adminStatusText.textContent = "No se pudo cargar el dataset global; el modo local sigue disponible.";
  }
}

async function loginAdmin() {
  try {
    const response = await fetch("/api/admin-login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        username: adminUserInput.value.trim(),
        password: adminPasswordInput.value
      })
    });
    const data = await response.json();
    if (!response.ok) {
      adminStatusText.textContent = data.error ?? "No se pudo iniciar sesion";
      return;
    }

    adminToken = data.token;
    sessionStorage.setItem(ADMIN_TOKEN_KEY, adminToken);
    adminUserInput.value = "";
    adminPasswordInput.value = "";
    updateAdminUi();
  } catch (error) {
    console.error(error);
    adminStatusText.textContent = "No se pudo conectar con el servidor de administracion";
  }
}

async function saveGlobalItem(type, item) {
  try {
    const response = await fetch("/api/global-data", {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${adminToken}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ type, item })
    });
    const data = await response.json();
    if (!response.ok) {
      const counter = type === "letters" ? letterSampleCounter : sampleCounter;
      counter.textContent = data.error ?? "No se pudo guardar globalmente";
      return false;
    }

    globalSigns = Array.isArray(data.signs) ? data.signs : [];
    globalLetters = Array.isArray(data.letters) ? data.letters : [];
    return true;
  } catch (error) {
    console.error(error);
    const counter = type === "letters" ? letterSampleCounter : sampleCounter;
    counter.textContent = "No se pudo conectar para guardar globalmente";
    return false;
  }
}

async function deleteGlobalItem(type, label) {
  try {
    const response = await fetch(`/api/global-data?type=${type}&label=${encodeURIComponent(label)}`, {
      method: "DELETE",
      headers: { "Authorization": `Bearer ${adminToken}` }
    });
    const data = await response.json();
    if (!response.ok) {
      adminStatusText.textContent = data.error ?? "No se pudo borrar globalmente";
      return;
    }
    globalSigns = Array.isArray(data.signs) ? data.signs : [];
    globalLetters = Array.isArray(data.letters) ? data.letters : [];
    renderSignList();
    renderLetterList();
  } catch (error) {
    console.error(error);
    adminStatusText.textContent = "No se pudo conectar para borrar globalmente";
  }
}

function updateAdminUi() {
  const isAdmin = Boolean(adminToken);
  adminLoginPanel.classList.toggle("hidden", isAdmin);
  adminToggleButton.textContent = isAdmin ? "Salir de administrador" : "Inicio de sesion administrador";
  adminStatusText.textContent = isAdmin
    ? "Modo administrador: las nuevas muestras se guardan globalmente."
    : "Modo local: tus cambios solo quedan en esta computadora.";
  signScopeText.textContent = isAdmin ? "Guardado global" : "Guardado local";
  letterScopeText.textContent = isAdmin ? "Guardado global" : "Guardado local";
}

function renderPendingSamples(type) {
  const list = type === "letter" ? pendingLetterSampleList : pendingSampleList;
  const samples = type === "letter" ? pendingLetterSamples : pendingSamples;
  const counter = type === "letter" ? letterSampleCounter : sampleCounter;
  if (!list) return;

  list.innerHTML = "";
  if (samples.length === 0) return;

  samples.forEach((sample, index) => {
    const item = document.createElement("div");
    item.className = "sample-chip";
    const kind = sample.kind === "motion" ? "movimiento" : "pose";
    item.innerHTML = `<span>Muestra ${index + 1}: ${kind}</span>`;
    const removeButton = document.createElement("button");
    removeButton.type = "button";
    removeButton.textContent = "Quitar";
    removeButton.addEventListener("click", () => {
      samples.splice(index, 1);
      counter.textContent = type === "letter"
        ? `${samples.length} muestras de letra`
        : `${samples.length} muestras listas`;
      renderPendingSamples(type);
    });
    item.append(removeButton);
    list.append(item);
  });
}

function renderSignList() {
  signList.innerHTML = "";
  const allSigns = [
    ...globalSigns.map((item) => ({ ...item, scope: "global" })),
    ...customSigns.map((item) => ({ ...item, scope: "local" }))
  ];
  const canSearch = allSigns.length > SEARCH_ENABLE_THRESHOLD;
  signSearchInput.disabled = !canSearch;
  if (!canSearch) signSearchInput.value = "";

  if (allSigns.length === 0) {
    signList.innerHTML = '<p class="helper">Aun no hay senias ecuatorianas guardadas.</p>';
    return;
  }

  const query = signSearchInput.value.trim().toLowerCase();
  const visibleSigns = query
    ? allSigns.filter((sign) => sign.label.toLowerCase().includes(query))
    : allSigns;

  if (visibleSigns.length === 0) {
    signList.innerHTML = '<p class="helper">No hay senias que coincidan con la busqueda.</p>';
    return;
  }

  for (const sign of visibleSigns) {
    const chip = document.createElement("div");
    chip.className = "sign-chip";
    chip.innerHTML = `<span>${sign.label} (${sign.samples.length}) <small>${sign.scope}</small></span>`;
    const removeButton = document.createElement("button");
    removeButton.type = "button";
    removeButton.textContent = "Borrar";
    removeButton.addEventListener("click", () => {
      if (sign.scope === "global") {
        if (adminToken) deleteGlobalItem("signs", sign.label);
        return;
      }
      customSigns = customSigns.filter((item) => item.label !== sign.label);
      saveSigns(customSigns);
      renderSignList();
    });
    removeButton.disabled = sign.scope === "global" && !adminToken;
    chip.append(removeButton);
    signList.append(chip);
  }
}

function renderLetterSelect() {
  letterSelect.innerHTML = "";
  for (const letter of ALPHABET) {
    const option = document.createElement("option");
    option.value = letter;
    option.textContent = letter;
    letterSelect.append(option);
  }
}

function renderLetterList() {
  letterList.innerHTML = "";
  const allLetters = [
    ...globalLetters.map((item) => ({ ...item, scope: "global" })),
    ...customLetters.map((item) => ({ ...item, scope: "local" }))
  ];
  const canSearch = allLetters.length > SEARCH_ENABLE_THRESHOLD;
  letterSearchInput.disabled = !canSearch;
  if (!canSearch) letterSearchInput.value = "";

  if (allLetters.length === 0) {
    letterList.innerHTML = '<p class="helper">Puedes mejorar cada letra capturando tus propias muestras.</p>';
    return;
  }

  const query = letterSearchInput.value.trim().toLowerCase();
  const visibleLetters = query
    ? allLetters.filter((letter) => letter.label.toLowerCase().includes(query))
    : allLetters;

  if (visibleLetters.length === 0) {
    letterList.innerHTML = '<p class="helper">No hay letras que coincidan con la busqueda.</p>';
    return;
  }

  for (const letter of visibleLetters) {
    const chip = document.createElement("div");
    chip.className = "sign-chip";
    chip.innerHTML = `<span>${letter.label} (${letter.samples.length}) <small>${letter.scope}</small></span>`;
    const removeButton = document.createElement("button");
    removeButton.type = "button";
    removeButton.textContent = "Borrar";
    removeButton.addEventListener("click", () => {
      if (letter.scope === "global") {
        if (adminToken) deleteGlobalItem("letters", letter.label);
        return;
      }
      customLetters = customLetters.filter((item) => item.label !== letter.label);
      saveLetters(customLetters);
      renderLetterList();
    });
    removeButton.disabled = letter.scope === "global" && !adminToken;
    chip.append(removeButton);
    letterList.append(chip);
  }
}

function loadSigns() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY)) ?? [];
  } catch {
    return [];
  }
}

function saveSigns(signs) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(signs));
}

function loadLetters() {
  try {
    return JSON.parse(localStorage.getItem(LETTER_STORAGE_KEY)) ?? [];
  } catch {
    return [];
  }
}

function saveLetters(letters) {
  localStorage.setItem(LETTER_STORAGE_KEY, JSON.stringify(letters));
}
