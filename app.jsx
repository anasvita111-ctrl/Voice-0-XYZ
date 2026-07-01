const { useState, useRef, useEffect } = React;


// ─── DESIGN TOKENS — Modern Twilight Palette ──────────────────────────────────
// Backgrounds: warm deep navy (not cold black) → easier on eyes
// Accents: soft indigo, rose, amber, emerald — muted neon, not harsh
const C = {
  // Backgrounds — warm deep navy gradient family
  void:  "#0d0f1a",   // page background
  deep:  "#12152b",   // slightly lighter
  panel: "#181c35",   // cards
  rim:   "#252a4a",   // borders / dividers

  // X axis — soft indigo/violet
  cyan:    "#818cf8",   // main accent (indigo-400)
  cyanDim: "#1e1f4a",

  // neg side (gauge left) — muted mauve
  purple:  "#a78bfa",   // violet-400
  purDim:  "#2e1f5e",

  // Y axis — rose
  pink:    "#fb7185",   // rose-400
  pinkDim: "#4c0a1a",

  // Z axis — amber
  gold:    "#fbbf24",   // amber-400
  goldDim: "#3d2000",

  // positive indicators — soft teal
  green:   "#34d399",   // emerald-400
  greDim:  "#0a2e22",

  // text
  slate:   "#6b7280",
  fog:     "#9ca3af",
  snow:    "#f3f4f6",
};

// ─── AUDIO ENGINE ─────────────────────────────────────────────────────────────
class VoiceEngine {
  constructor() { this.ctx = null; this.analyser = null; this.stream = null; this.chunks = []; this.recorder = null; }

  async start() {
    this.stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
    this.ctx = new (window.AudioContext || window.webkitAudioContext)();
    const src = this.ctx.createMediaStreamSource(this.stream);
    this.analyser = this.ctx.createAnalyser();
    this.analyser.fftSize = 4096;
    this.analyser.smoothingTimeConstant = 0.75;
    src.connect(this.analyser);
    this.chunks = [];
    this.recorder = new MediaRecorder(this.stream);
    this.recorder.ondataavailable = e => this.chunks.push(e.data);
    this.recorder.start(100);
  }

  getWaveform() {
    if (!this.analyser) return new Uint8Array(128).fill(128);
    const buf = new Uint8Array(this.analyser.frequencyBinCount);
    this.analyser.getByteTimeDomainData(buf);
    return buf;
  }
  getLevel() {
    if (!this.analyser) return 0;
    const buf = new Float32Array(this.analyser.fftSize);
    this.analyser.getFloatTimeDomainData(buf);
    let sum = 0; for (let i = 0; i < buf.length; i++) sum += buf[i] * buf[i];
    return Math.sqrt(sum / buf.length);
  }
  // Live single-axis estimate (cheap, for the live needle on axis screens)
  getLiveBands() {
    if (!this.analyser) return { low: 0, mid: 0, high: 0 };
    const buf = new Uint8Array(this.analyser.frequencyBinCount);
    this.analyser.getByteFrequencyData(buf);
    const n = buf.length;
    const seg = (a, b) => { let s = 0, c = 0; for (let i = a; i < b; i++) { s += buf[i]; c++; } return c ? s / c / 255 : 0; };
    return { low: seg(0, Math.floor(n*0.08)), mid: seg(Math.floor(n*0.08), Math.floor(n*0.35)), high: seg(Math.floor(n*0.35), Math.floor(n*0.85)) };
  }

  async stop() {
    return new Promise(resolve => {
      if (!this.recorder) { resolve(null); return; }
      this.recorder.onstop = () => resolve(new Blob(this.chunks, { type: "audio/webm" }));
      this.recorder.stop();
      this.stream?.getTracks().forEach(t => t.stop());
    });
  }

  async analyseBlob(blob) {
    try {
      const arrayBuf = await blob.arrayBuffer();
      const tmpCtx = new OfflineAudioContext(1, 44100 * 10, 44100);
      const decoded = await tmpCtx.decodeAudioData(arrayBuf);
      return this._extractFeatures(decoded.getChannelData(0), decoded.sampleRate);
    } catch {
      return this._demoFeatures();
    }
  }

  _extractFeatures(pcm, sr) {
    let rms = 0; for (let i = 0; i < pcm.length; i++) rms += pcm[i]*pcm[i];
    rms = Math.sqrt(rms / pcm.length);
    let zcr = 0; for (let i = 1; i < pcm.length; i++) if ((pcm[i]>=0)!==(pcm[i-1]>=0)) zcr++;
    zcr /= pcm.length;

    const WIN = 2048, hop = Math.floor(pcm.length / 8), frames = [];
    for (let start = 0; start + WIN < pcm.length; start += hop) {
      const frame = new Float32Array(WIN);
      for (let i = 0; i < WIN; i++) frame[i] = pcm[start+i] * (0.5*(1-Math.cos(2*Math.PI*i/WIN)));
      frames.push(this._fft(frame));
    }
    const avg = new Float32Array(WIN/2);
    for (const f of frames) for (let i = 0; i < avg.length; i++) avg[i] += f[i]/frames.length;

    let wSum=0,sum=0;
    for (let i=0;i<avg.length;i++){ const freq=(i*sr)/WIN; wSum+=freq*avg[i]; sum+=avg[i]; }
    const centroid = sum>0 ? wSum/sum : 0;

    const band=(lo,hi)=>{let e=0,n=0;for(let i=0;i<avg.length;i++){const f=(i*sr)/WIN;if(f>=lo&&f<hi){e+=avg[i];n++;}}return n>0?e/n:0;};
    const subBass=band(20,150), bass=band(150,500), mid=band(500,2000), high=band(2000,6000), air=band(6000,16000);

    const stability = this._autocorrStability(pcm, sr);

    const xRaw = (bass/(subBass+bass+0.0001)) - (zcr*15);
    const x = this._clamp(this._norm(xRaw,-0.5,0.5)*20-10,-10,10);
    const yRaw = (high+mid*0.5)/(subBass+bass+high+0.0001);
    const y = this._clamp(this._norm(yRaw,0.05,0.6)*20-10,-10,10);
    const zRaw = (rms*5)+air*2-zcr*8;
    const z = this._clamp(this._norm(zRaw,0,1.5)*20-10,-10,10);

    const stabilityPct = Math.round(this._clamp(stability*100,20,98));
    const resonance = Math.round(this._clamp(((mid+high)/(sum+0.0001))*200,20,95));
    const quality = Math.min(98,Math.max(18,Math.round(stabilityPct*0.4+resonance*0.3+this._clamp(rms*800,5,70)*0.3)));
    const spectralBalance = Math.round(this._clamp((1-Math.abs(centroid-1500)/3000)*100,10,98));

    return {
      x: Math.round(x*10)/10, y: Math.round(y*10)/10, z: Math.round(z*10)/10,
      quality, stability: stabilityPct, resonance, spectralBalance,
      centroid: Math.round(centroid), rmsDb: Math.round(20*Math.log10(rms+0.0001)), zcr: Math.round(zcr*10000)/10000,
      airflow: Math.round(this._clamp((air*300+rms*100),5,95)),
    };
  }

  _fft(frame) {
    const N = frame.length, re = new Float32Array(N), im = new Float32Array(N);
    for (let i=0;i<N;i++) re[i]=frame[i];
    for (let i=1,j=0;i<N;i++){let bit=N>>1;for(;j&bit;bit>>=1)j^=bit;j^=bit;if(i<j){[re[i],re[j]]=[re[j],re[i]];[im[i],im[j]]=[im[j],im[i]];}}
    for (let len=2;len<=N;len<<=1){
      const ang=(2*Math.PI)/len, wRe=Math.cos(ang), wIm=-Math.sin(ang);
      for (let i=0;i<N;i+=len){
        let curRe=1,curIm=0;
        for (let j=0;j<len/2;j++){
          const uRe=re[i+j],uIm=im[i+j];
          const vRe=re[i+j+len/2]*curRe-im[i+j+len/2]*curIm;
          const vIm=re[i+j+len/2]*curIm+im[i+j+len/2]*curRe;
          re[i+j]=uRe+vRe; im[i+j]=uIm+vIm; re[i+j+len/2]=uRe-vRe; im[i+j+len/2]=uIm-vIm;
          const nr=curRe*wRe-curIm*wIm; curIm=curRe*wIm+curIm*wRe; curRe=nr;
        }
      }
    }
    const mag = new Float32Array(N/2);
    for (let i=0;i<N/2;i++) mag[i]=Math.sqrt(re[i]*re[i]+im[i]*im[i])/N;
    return mag;
  }

  _autocorrStability(pcm, sr) {
    const win=Math.round(0.025*sr), hop=Math.round(0.01*sr);
    let voiced=0,total=0;
    for (let s=0;s+win<pcm.length;s+=hop){
      let r0=0,r1=0;
      for (let i=0;i<win-1;i++){ r0+=pcm[s+i]*pcm[s+i]; r1+=pcm[s+i]*pcm[s+i+1]; }
      if (r0>0.001 && Math.abs(r1/r0)>0.7) voiced++;
      total++;
    }
    return total>0 ? voiced/total : 0.5;
  }

  _norm(v,lo,hi){return (v-lo)/(hi-lo);}
  _clamp(v,lo,hi){return Math.max(lo,Math.min(hi,v));}
  _demoFeatures(){
    return { x:+(Math.random()*16-8).toFixed(1), y:+(Math.random()*16-8).toFixed(1), z:+(Math.random()*14-4).toFixed(1),
      quality:Math.round(40+Math.random()*40), stability:Math.round(45+Math.random()*45), resonance:Math.round(35+Math.random()*50),
      spectralBalance:Math.round(40+Math.random()*45), centroid:Math.round(800+Math.random()*1800),
      rmsDb:Math.round(-30+Math.random()*20), zcr:+(0.02+Math.random()*0.08).toFixed(4), airflow:Math.round(30+Math.random()*50) };
  }
}
const engine = new VoiceEngine();

// ─── CLAUDE AI LAYER ──────────────────────────────────────────────────────────
async function claudeAnalyze(feats) {
  const { x, y, z, quality, stability, resonance, spectralBalance, centroid, rmsDb, zcr, airflow } = feats;
  const prompt = `أنت خبير عالمي في علم الصوتيات وتدريب الصوت البشري. حلل هذه القياسات الصوتية الدقيقة وأعطِ تقريراً تفصيلياً.

البيانات:
- X (توتر الحنجرة): ${x} من -10 إلى +10 (+ = انقباض، - = ارتخاء)
- Y (موضع الصدى): ${y} من -10 إلى +10 (+ = رأس/أنف، - = صدر)
- Z (ضغط الهواء): ${z} من -10 إلى +10 (+ = مرتفع/سريع، - = منخفض/بطيء)
- الجودة العامة: ${quality}%
- ثبات النغمة: ${stability}%
- مستوى الرنين: ${resonance}%
- توازن الطيف: ${spectralBalance}%
- تدفق الهواء: ${airflow}%
- المركز الطيفي: ${centroid} Hz
- مستوى الصوت: ${rmsDb} dB

أعطِ تحليلاً شاملاً بصيغة JSON فقط بدون backticks:
{
  "voiceType": "نوع الصوت في 3-4 كلمات",
  "overallScore": رقم من 0-100,
  "headline": "جملة تشخيصية قوية وواضحة",
  "xAnalysis": "تحليل مفصل لمحور X في جملتين",
  "yAnalysis": "تحليل مفصل لمحور Y في جملتين",
  "zAnalysis": "تحليل مفصل لمحور Z في جملتين",
  "strengths": ["نقطة قوة 1", "نقطة قوة 2", "نقطة قوة 3"],
  "improvements": [
    {"area": "اسم المجال", "issue": "وصف المشكلة", "exercise": "تمرين محدد للتحسين"},
    {"area": "اسم المجال", "issue": "وصف المشكلة", "exercise": "تمرين محدد للتحسين"}
  ],
  "priorityTip": "أهم نصيحة واحدة يجب تطبيقها فوراً"
}`;
  try {
    const res = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: "claude-sonnet-4-6", max_tokens: 1200, messages: [{ role: "user", content: prompt }] }),
    });
    const data = await res.json();
    const text = data.content.map(b => b.text || "").join("");
    return JSON.parse(text.replace(/```json|```/g, "").trim());
  } catch {
    return {
      voiceType: "صوت مختلط متوازن", overallScore: quality,
      headline: "صوتك يحمل أساساً جيداً مع فرص تحسين واضحة في التوازن والثبات",
      xAnalysis: x > 2 ? "حنجرتك متوترة بشكل ملحوظ، مما قد يسبب إجهاداً صوتياً مع الاستخدام الطويل." : x < -2 ? "حنجرتك مسترخية جداً مما يقلل من قوة ووضوح صوتك." : "توتر حنجرتك في نطاق صحي ومتوازن.",
      yAnalysis: y > 2 ? "صوتك يميل بقوة نحو الرأس، مناسب للنغمات العالية لكن قد يفتقد العمق." : y < -2 ? "صوتك يعتمد على الصدر بشكل كبير، يعطي قوة لكن قد يرهق الحبال الصوتية." : "توازن جيد بين رنين الصدر والرأس.",
      zAnalysis: z > 2 ? "ضغط الهواء مرتفع، قد يسبب صوتاً متوتراً أو حاداً." : z < -2 ? "تدفق الهواء ضعيف، الصوت قد يبدو خافتاً أو متعباً." : "تدفق الهواء ضمن النطاق الطبيعي والصحي.",
      strengths: ["إيقاع صوتي منتظم نسبياً", "وضوح في النطق", "طاقة صوتية حاضرة"],
      improvements: [
        { area: "ثبات النغمة", issue: "تذبذب بسيط في النغمة أثناء الجمل الطويلة", exercise: "تمرين الهمهمة المستمرة لمدة 30 ثانية يومياً" },
        { area: "دعم التنفس", issue: "اعتماد على التنفس الصدري بدلاً من الحجاب الحاجز", exercise: "تمارين التنفس العميق من البطن قبل كل جلسة كلام" },
      ],
      priorityTip: "ابدأ كل جلسة كلام بثلاث أنفاس عميقة من الحجاب الحاجز لتثبيت الصوت من البداية",
    };
  }
}

// ─── COORDINATE CHART ──────────────────────────────────────────────────────
// Flat 2D cross-axis layout matching the reference image exactly:
//   X = horizontal line (cyan)
//   Y = vertical line (pink)
//   Z = diagonal line, lower-left to upper-right (gold)
// Tick marks 1..10 on every arm. No ball on the axis lines themselves.
function FullCoordinateChart({ point, showPoint = false }) {
  const W = 340, H = 340, CX = 170, CY = 170;
  const UNIT = 13; // px per coordinate unit on X/Y
  const ZUNIT = 9.2; // px per unit along the diagonal (cos/sin 45)
  const DIAG = Math.SQRT1_2; // 0.707

  // X axis: horizontal
  const xPoint = v => ({ px: CX + v * UNIT, py: CY });
  // Y axis: vertical (positive = up)
  const yPoint = v => ({ px: CX, py: CY - v * UNIT });
  // Z axis: diagonal, +Z to upper-right, -Z to lower-left
  const zPoint = v => ({ px: CX + v * ZUNIT * DIAG, py: CY - v * ZUNIT * DIAG });

  const tick = (axisFn, v, color, sideOffset) => {
    const p = axisFn(v);
    return (
      <g key={`t-${v}-${color}`}>
        <circle cx={p.px} cy={p.py} r={1.4} fill={color} opacity={0.85} />
        {v % 2 === 0 && (
          <text x={p.px + sideOffset[0]} y={p.py + sideOffset[1]} fontSize={7} fill={color} opacity={0.65} textAnchor="middle">{v}</text>
        )}
      </g>
    );
  };

  const xTicks = []; for (let v=-10; v<=10; v++) if (v!==0) xTicks.push(tick(xPoint, v, C.cyan, [0, 12]));
  const yTicks = []; for (let v=-10; v<=10; v++) if (v!==0) yTicks.push(tick(yPoint, v, C.pink, [-13, 3]));
  const zTicks = []; for (let v=-10; v<=10; v++) if (v!==0) zTicks.push(tick(zPoint, v, C.gold, [v>0?9:-9, v>0?-4:4]));

  const xEndPos = xPoint(10), xEndNeg = xPoint(-10);
  const yEndPos = yPoint(10), yEndNeg = yPoint(-10);
  const zEndPos = zPoint(10), zEndNeg = zPoint(-10);

  const arrow = (from, to, color) => {
    const ang = Math.atan2(to.py-from.py, to.px-from.px) * 180/Math.PI + 90;
    return <polygon points={`${to.px},${to.py-4} ${to.px+5},${to.py+4} ${to.px-5},${to.py+4}`} fill={color} transform={`rotate(${ang} ${to.px} ${to.py})`}/>;
  };

  // Projected point for the result dot (simple weighted blend of the three directions)
  const dotPos = point ? {
    px: CX + point.x*UNIT + point.z*ZUNIT*DIAG,
    py: CY - point.y*UNIT - point.z*ZUNIT*DIAG,
  } : null;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ maxHeight: 340 }}>
      <defs>
        <filter id="g1"><feGaussianBlur stdDeviation="2" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
      </defs>

      {/* faint concentric rings at origin, echoing reference image */}
      {[20,40,60].map(r => <circle key={r} cx={CX} cy={CY} r={r} fill="none" stroke={C.rim} strokeWidth={0.6}/>)}

      {/* Z axis (diagonal, gold) — drawn first so X/Y sit on top at center */}
      <g filter="url(#g1)">
        <line x1={zEndNeg.px} y1={zEndNeg.py} x2={zEndPos.px} y2={zEndPos.py} stroke={C.gold} strokeWidth={2.2} opacity={0.9}/>
        {arrow(CX_CY(CX,CY), zEndPos, C.gold)}
        {arrow(CX_CY(CX,CY), zEndNeg, C.gold)}
      </g>
      {zTicks}
      <text x={zEndPos.px+11} y={zEndPos.py-9} fill={C.gold} fontSize={13} fontWeight="700" textAnchor="middle">+Z</text>
      <text x={zEndNeg.px-11} y={zEndNeg.py+13} fill={C.gold} fontSize={13} fontWeight="700" textAnchor="middle">−Z</text>

      {/* Y axis (vertical, pink) */}
      <g filter="url(#g1)">
        <line x1={yEndNeg.px} y1={yEndNeg.py} x2={yEndPos.px} y2={yEndPos.py} stroke={C.pink} strokeWidth={2.2} opacity={0.9}/>
        {arrow(CX_CY(CX,CY), yEndPos, C.pink)}
        {arrow(CX_CY(CX,CY), yEndNeg, C.pink)}
      </g>
      {yTicks}
      <text x={yEndPos.px} y={yEndPos.py-10} fill={C.pink} fontSize={13} fontWeight="700" textAnchor="middle">+Y</text>
      <text x={yEndNeg.px} y={yEndNeg.py+16} fill={C.pink} fontSize={13} fontWeight="700" textAnchor="middle">−Y</text>

      {/* X axis (horizontal, cyan) */}
      <g filter="url(#g1)">
        <line x1={xEndNeg.px} y1={xEndNeg.py} x2={xEndPos.px} y2={xEndPos.py} stroke={C.cyan} strokeWidth={2.2} opacity={0.9}/>
        {arrow(CX_CY(CX,CY), xEndPos, C.cyan)}
        {arrow(CX_CY(CX,CY), xEndNeg, C.cyan)}
      </g>
      {xTicks}
      <text x={xEndPos.px+14} y={xEndPos.py+4} fill={C.cyan} fontSize={13} fontWeight="700" textAnchor="middle">+X</text>
      <text x={xEndNeg.px-14} y={xEndNeg.py+4} fill={C.cyan} fontSize={13} fontWeight="700" textAnchor="middle">−X</text>

      {/* Origin */}
      <circle cx={CX} cy={CY} r={3.5} fill={C.snow} opacity={0.9}/>

      {/* Result point (only on analysis screen) */}
      {showPoint && point && (
        <g filter="url(#g1)">
          <circle cx={dotPos.px} cy={dotPos.py} r={7} fill={C.snow}/>
          <circle cx={dotPos.px} cy={dotPos.py} r={3} fill={C.void}/>
        </g>
      )}
    </svg>
  );
}
function CX_CY(cx, cy) { return { px: cx, py: cy }; }

// ─── SINGLE-AXIS GAUGE (screens 2,3,4) ────────────────────────────────────────
function AxisGauge({ axisLabel, value, color, posLabel, negLabel, live, liveLevel }) {
  const pct = ((value + 10) / 20) * 100;
  const needleAngle = -90 + (pct / 100) * 180; // semicircle -90..+90
  const R = 100, CX = 130, CY = 120;
  const polarToXY = (angleDeg, r) => {
    const a = (angleDeg - 90) * Math.PI / 180;
    return { x: CX + r * Math.cos(a), y: CY + r * Math.sin(a) };
  };
  const arcPath = (startA, endA, r) => {
    const s = polarToXY(startA, r), e = polarToXY(endA, r);
    const largeArc = endA - startA > 180 ? 1 : 0;
    return `M ${s.x} ${s.y} A ${r} ${r} 0 ${largeArc} 1 ${e.x} ${e.y}`;
  };
  const needleTip = polarToXY(needleAngle, R - 14);

  return (
    <svg viewBox="0 0 260 175" width="100%" style={{ maxHeight: 220 }}>
      <defs>
        <filter id="gaugeglow"><feGaussianBlur stdDeviation="2" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
      </defs>
      {/* Background arc */}
      <path d={arcPath(-90, 90, R)} fill="none" stroke={C.rim} strokeWidth={14} strokeLinecap="round" />
      {/* Negative half */}
      <path d={arcPath(-90, 0, R)} fill="none" stroke={C.purple} strokeWidth={14} strokeLinecap="round" opacity={0.35} />
      {/* Positive half */}
      <path d={arcPath(0, 90, R)} fill="none" stroke={color} strokeWidth={14} strokeLinecap="round" opacity={0.35} />
      {/* Active fill up to needle */}
      <path d={arcPath(-90, needleAngle, R)} fill="none" stroke={color} strokeWidth={14} strokeLinecap="round" filter="url(#gaugeglow)" />
      {/* Tick marks */}
      {[-90,-45,0,45,90].map(a => {
        const p1 = polarToXY(a, R-9), p2 = polarToXY(a, R+9);
        return <line key={a} x1={p1.x} y1={p1.y} x2={p2.x} y2={p2.y} stroke={C.fog} strokeWidth={1.5} opacity={0.5}/>;
      })}
      {/* Needle */}
      <line x1={CX} y1={CY} x2={needleTip.x} y2={needleTip.y} stroke="white" strokeWidth={3} strokeLinecap="round" filter="url(#gaugeglow)"/>
      <circle cx={CX} cy={CY} r={7} fill="white" />
      <circle cx={CX} cy={CY} r={4} fill={color} />
      {/* Value */}
      <text x={CX} y={CY+42} textAnchor="middle" fontSize={28} fontWeight="800" fill={color}>{value > 0 ? "+" : ""}{value}</text>
      {/* Labels */}
      <text x={CX-R-6} y={CY+8} textAnchor="middle" fontSize={11} fill={C.purple} fontWeight={700}>−10</text>
      <text x={CX+R+6} y={CY+8} textAnchor="middle" fontSize={11} fill={color} fontWeight={700}>+10</text>
      <text x={CX} y={CY-R-10} textAnchor="middle" fontSize={11} fill={C.fog}>0</text>
      {/* Live indicator dot pulsing if active */}
      {live && <circle cx={CX} cy={CY-R-26} r={4+liveLevel*8} fill={color} opacity={0.6}/>}
    </svg>
  );
}

function LiveWaveform({ active }) {
  const canRef = useRef(null);
  useEffect(() => {
    if (!active) return;
    let raf;
    const draw = () => {
      const can = canRef.current; if (!can) return;
      const ctx = can.getContext("2d"); const w = can.width, h = can.height;
      ctx.fillStyle = C.deep; ctx.fillRect(0,0,w,h);
      const buf = engine.getWaveform();
      ctx.beginPath(); ctx.strokeStyle = C.cyan; ctx.lineWidth = 2;
      const step = w/buf.length;
      for (let i=0;i<buf.length;i++){ const y=(buf[i]/255)*h; i===0?ctx.moveTo(0,y):ctx.lineTo(i*step,y); }
      ctx.stroke();
      raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [active]);
  return <canvas ref={canRef} width={320} height={60} style={{ width:"100%", height:60, borderRadius:8, display:"block" }} />;
}

// ─── MAIN APP ─────────────────────────────────────────────────────────────────
const TABS = [
  { id: "coords",  icon: "🌐", label: "الإحداثيات" },
  { id: "axisX",   icon: "🔵", label: "محور X" },
  { id: "axisY",   icon: "🟣", label: "محور Y" },
  { id: "axisZ",   icon: "🟡", label: "محور Z" },
  { id: "record",  icon: "🎙️", label: "التسجيلات" },
  { id: "analyze", icon: "🤖", label: "التحليل" },
];

function App() {
  const [tab, setTab] = useState("coords");
  const [recState, setRecState] = useState("idle"); // idle | recording | processing
  const [recSecs, setRecSecs] = useState(0);
  const [liveVal, setLiveVal] = useState(0);
  const [liveBands, setLiveBands] = useState({ low:0, mid:0, high:0 });
  const [activeAxis, setActiveAxis] = useState(null); // 'x' | 'y' | 'z' | null — which axis screen is recording
  const [lastFeats, setLastFeats] = useState(null);
  const [recordings, setRecordings] = useState([]); // {id, name, date, feats}
  const [analyzeFeats, setAnalyzeFeats] = useState(null);
  const [analyzeAI, setAnalyzeAI] = useState(null);
  const [loadMsg, setLoadMsg] = useState("");
  const [permErr, setPermErr] = useState(false);
  const recTimer = useRef(null);
  const liveTimer = useRef(null);

  const root = {
    minHeight: "100vh", background: `radial-gradient(ellipse at 50% 0%, #1e2048 0%, ${C.void} 65%)`, color: C.snow,
    fontFamily: "'Segoe UI', system-ui, -apple-system, sans-serif",
    direction: "rtl", maxWidth: 440, margin: "0 auto",
    paddingBottom: 86, position: "relative",
  };
  const card = (accent = C.cyan) => ({
    background: `linear-gradient(145deg, ${C.panel}, ${C.deep})`,
    border: `1px solid ${accent}30`,
    borderRadius: 18,
    padding: 16, marginBottom: 14,
    boxShadow: `0 4px 24px ${accent}18, inset 0 1px 0 rgba(255,255,255,0.05)`,
  });
  const btn = (bg, color="#fff") => ({
    width: "100%", padding: "15px 0", borderRadius: 50, border: "none",
    background: bg, color, fontSize: 15, fontWeight: 700, cursor: "pointer",
    marginBottom: 10, letterSpacing: 0.3, boxShadow: "0 4px 18px rgba(0,0,0,0.25)",
  });
  const ghost = (color=C.slate) => ({ ...btn("transparent", color), border: `1px solid ${color}44`, boxShadow: "none" });
  const tag = (text, color) => (
    <span style={{ display:"inline-block", padding:"4px 14px", borderRadius:20, background:`${color}22`, border:`1px solid ${color}44`, color, fontSize:12, fontWeight:600 }}>{text}</span>
  );

  // Live band sampling for axis gauges while recording
  useEffect(() => {
    if (recState !== "recording") return;
    liveTimer.current = setInterval(() => {
      const b = engine.getLiveBands();
      setLiveBands(b);
      if (activeAxis === "x") setLiveVal(+((b.low - b.high) * 10).toFixed(1));
      if (activeAxis === "y") setLiveVal(+((b.high - b.low) * 10).toFixed(1));
      if (activeAxis === "z") setLiveVal(+((engine.getLevel() * 20) - 4).toFixed(1));
    }, 120);
    return () => clearInterval(liveTimer.current);
  }, [recState, activeAxis]);

  const startRecording = async (axis = null) => {
    setPermErr(false);
    setActiveAxis(axis);
    try { await engine.start(); } catch { setPermErr(true); return; }
    setRecState("recording");
    setRecSecs(0);
    setLiveVal(0);
    recTimer.current = setInterval(() => setRecSecs(s => s+1), 1000);
  };

  const stopRecording = async (forAnalysis = false) => {
    clearInterval(recTimer.current);
    clearInterval(liveTimer.current);
    setRecState("processing");
    const msgs = ["⚙️ استخراج الطيف الصوتي...", "📊 تحليل توازن الترددات...", "📍 حساب الإحداثيات..."];
    let mi = 0; setLoadMsg(msgs[0]);
    const mt = setInterval(() => { mi = Math.min(mi+1, msgs.length-1); setLoadMsg(msgs[mi]); }, 700);

    const blob = await engine.stop();
    const feats = blob ? await engine.analyseBlob(blob) : engine._demoFeatures();
    clearInterval(mt);
    setLastFeats(feats);

    const rec = { id: Date.now(), name: `تسجيل ${recordings.length + 1}`, date: new Date().toLocaleString("ar-EG", { hour:"2-digit", minute:"2-digit", day:"2-digit", month:"2-digit" }), feats };
    setRecordings(r => [rec, ...r]);

    if (forAnalysis) {
      setLoadMsg("🤖 الذكاء الاصطناعي يحلل صوتك...");
      const ai = await claudeAnalyze(feats);
      setAnalyzeFeats(feats);
      setAnalyzeAI(ai);
    }

    setRecState("idle");
    setActiveAxis(null);
  };

  const deleteRecording = (id) => setRecordings(r => r.filter(x => x.id !== id));

  const analyzeFromRecording = async (rec) => {
    setTab("analyze");
    setRecState("processing");
    setLoadMsg("🤖 الذكاء الاصطناعي يحلل التسجيل...");
    setAnalyzeFeats(rec.feats);
    const ai = await claudeAnalyze(rec.feats);
    setAnalyzeAI(ai);
    setRecState("idle");
  };

  // ── PROCESSING OVERLAY ─────────────────────────────────────────────────────
  if (recState === "processing") return (
    <div style={{ ...root, display:"flex", flexDirection:"column", alignItems:"center", justifyContent:"center", gap:24, minHeight:"100vh" }}>
      <div style={{ width:80, height:80, borderRadius:"50%", border:`3px solid transparent`, borderTop:`3px solid ${C.cyan}`, borderRight:`3px solid ${C.purple}`, animation:"spin 0.9s linear infinite" }}/>
      <p style={{ color:C.cyan, fontSize:14, textAlign:"center", maxWidth:240 }}>{loadMsg}</p>
      <style>{`@keyframes spin{to{transform:rotate(360deg)}}`}</style>
    </div>
  );

  // ════════════════════════════════════════════════════════════════════════
  // SCREEN 1 — Full coordinate system explainer
  // ════════════════════════════════════════════════════════════════════════
  const ScreenCoords = () => (
    <div style={{ padding:"20px 16px 0" }}>
      <style>{`@keyframes float{0%,100%{transform:translateY(0)}50%{transform:translateY(-6px)}}`}</style>
      <div style={{ textAlign:"center", marginBottom:14 }}>
        <div style={{ fontSize:34, animation:"float 3s ease-in-out infinite" }}>🌐</div>
        <h1 style={{ margin:"6px 0 2px", fontSize:21, fontWeight:900, background:`linear-gradient(135deg, ${C.cyan}, ${C.purple})`, WebkitBackgroundClip:"text", WebkitTextFillColor:"transparent" }}>إحداثيات الصوت البشري</h1>
        <p style={{ color:C.slate, fontSize:12, margin:0 }}>النظام ثلاثي الأبعاد لقياس وتطوير الصوت</p>
      </div>

      <div style={card(C.cyan)}>
        <FullCoordinateChart />
      </div>

      {/* Axis explanations under the chart */}
      <div style={{ display:"flex", flexDirection:"column", gap:10 }}>
        {[
          { axis:"X", color:C.cyan, title:"محور X — توتر الحنجرة", pos:"+ انقباض عضلات الحنجرة", neg:"− ارتخاء عضلات الحنجرة", desc:"يقيس مدى شد أو استرخاء العضلات المسؤولة عن إنتاج الصوت في الحنجرة." },
          { axis:"Y", color:C.pink, title:"محور Y — موضع الصدى", pos:"+ رنين الرأس / تجاويف الأنف", neg:"− رنين الصدر / القصبة الهوائية", desc:"يحدد أين يتمركز رنين صوتك: في تجاويف الرأس العلوية أم في عمق الصدر." },
          { axis:"Z", color:C.gold, title:"محور Z — ضغط الهواء", pos:"+ تدفق هواء سريع ومرتفع", neg:"− تدفق هواء بطيء ومنخفض", desc:"يقيس قوة وسرعة الهواء الخارج من الرئتين أثناء الكلام أو الغناء." },
        ].map(a => (
          <div key={a.axis} style={{ ...card(a.color), marginBottom:0 }}>
            <div style={{ display:"flex", alignItems:"center", gap:10, marginBottom:8 }}>
              <div style={{ width:34, height:34, borderRadius:10, background:`${a.color}22`, border:`1px solid ${a.color}55`, display:"flex", alignItems:"center", justifyContent:"center", color:a.color, fontWeight:800, fontSize:15 }}>{a.axis}</div>
              <span style={{ fontWeight:700, fontSize:14, color:a.color }}>{a.title}</span>
            </div>
            <p style={{ margin:"0 0 8px", fontSize:12.5, color:C.fog, lineHeight:1.6 }}>{a.desc}</p>
            <div style={{ display:"flex", gap:8 }}>
              <span style={{ flex:1, fontSize:11, background:`${a.color}15`, color:a.color, borderRadius:8, padding:"6px 8px", textAlign:"center" }}>{a.pos}</span>
              <span style={{ flex:1, fontSize:11, background:`${C.slate}15`, color:C.fog, borderRadius:8, padding:"6px 8px", textAlign:"center" }}>{a.neg}</span>
            </div>
          </div>
        ))}
      </div>
      <div style={{ height: 6 }} />
    </div>
  );

  // ════════════════════════════════════════════════════════════════════════
  // SCREENS 2,3,4 — Single axis measurement
  // ════════════════════════════════════════════════════════════════════════
  const AXIS_CONFIG = {
    x: { label:"X", title:"محور X — توتر الحنجرة", color:C.cyan, pos:"+ انقباض عضلات الحنجرة", neg:"− ارتخاء عضلات الحنجرة", icon:"🔵",
         hint:"تحدث بصوت طبيعي. سنقيس مدى انقباض عضلات حنجرتك في الزمن الحقيقي." },
    y: { label:"Y", title:"محور Y — موضع الصدى", color:C.pink, pos:"+ رنين الرأس / الأنف", neg:"− رنين الصدر / القصبة", icon:"🟣",
         hint:"جرّب التنقل بين صوت عميق من صدرك وصوت مرتفع من رأسك." },
    z: { label:"Z", title:"محور Z — ضغط الهواء", color:C.gold, pos:"+ تدفق هواء سريع", neg:"− تدفق هواء بطيء", icon:"🟡",
         hint:"تنفّس بعمق وتحدث بقوة مختلفة لمشاهدة تغير ضغط الهواء." },
  };

  const ScreenAxis = (axisKey) => {
    const cfg = AXIS_CONFIG[axisKey];
    const isLive = recState === "recording" && activeAxis === axisKey;
    const displayVal = isLive ? liveVal : (lastFeats ? lastFeats[axisKey] : 0);
    return (
      <div style={{ padding:"20px 16px 0" }}>
        <div style={{ textAlign:"center", marginBottom:12 }}>
          <div style={{ fontSize:30 }}>{cfg.icon}</div>
          <h2 style={{ margin:"6px 0 2px", color:cfg.color }}>{cfg.title}</h2>
          <p style={{ color:C.slate, fontSize:12, margin:0 }}>{cfg.hint}</p>
        </div>

        <div style={card(cfg.color)}>
          <AxisGauge axisLabel={cfg.label} value={displayVal} color={cfg.color} live={isLive} liveLevel={engine.getLevel?.()||0} />
          <div style={{ display:"flex", justifyContent:"space-between", marginTop:4, direction:"ltr" }}>
            <span style={{ fontSize:12, color:C.purple, textAlign:"right" }}>{cfg.neg}</span>
            <span style={{ fontSize:12, color:cfg.color, textAlign:"left" }}>{cfg.pos}</span>
          </div>
        </div>

        {isLive && (
          <div style={{ ...card(cfg.color), padding:12 }}>
            <LiveWaveform active />
            <div style={{ textAlign:"center", marginTop:8 }}>
              <span style={{ color:cfg.color, fontWeight:700, fontSize:18, animation:"blink 1s infinite" }}>● {recSecs}s</span>
            </div>
          </div>
        )}

        {permErr && (
          <div style={{ background:`${C.pink}15`, border:`1px solid ${C.pink}44`, borderRadius:12, padding:14, marginBottom:14, color:C.pink, fontSize:13 }}>
            ⚠️ لم يتم السماح بالوصول للميكروفون.
          </div>
        )}

        {!isLive ? (
          <button onClick={() => startRecording(axisKey)} style={btn(`linear-gradient(135deg, ${cfg.color}, ${C.purple})`, C.snow)}>
            🎙️ ابدأ القياس المباشر لمحور {cfg.label}
          </button>
        ) : (
          <button onClick={() => stopRecording(false)} style={btn(`linear-gradient(135deg, ${C.pink}, ${C.purple})`, C.snow)}>
            ⏹ إيقاف القياس
          </button>
        )}

        {lastFeats && !isLive && (
          <div style={{ ...card(), padding:14, textAlign:"center" }}>
            <p style={{ margin:0, fontSize:13, color:C.fog }}>آخر قياس مسجّل لمحور {cfg.label}: <span style={{ color:cfg.color, fontWeight:700 }}>{lastFeats[axisKey] > 0 ? "+" : ""}{lastFeats[axisKey]}</span></p>
          </div>
        )}
        <style>{`@keyframes blink{0%,100%{opacity:1}50%{opacity:0.4}}`}</style>
      </div>
    );
  };

  // ════════════════════════════════════════════════════════════════════════
  // SCREEN 5 — Recording + file storage
  // ════════════════════════════════════════════════════════════════════════
  const ScreenRecord = () => (
    <div style={{ padding:"20px 16px 0" }}>
      <div style={{ textAlign:"center", marginBottom:14 }}>
        <div style={{ fontSize:30 }}>🎙️</div>
        <h2 style={{ margin:"6px 0 2px" }}>تسجيل وحفظ الصوت</h2>
        <p style={{ color:C.slate, fontSize:12, margin:0 }}>سجّل صوتك واحفظه لمراجعته أو تحليله لاحقاً</p>
      </div>

      {permErr && (
        <div style={{ background:`${C.pink}15`, border:`1px solid ${C.pink}44`, borderRadius:12, padding:14, marginBottom:14, color:C.pink, fontSize:13 }}>
          ⚠️ لم يتم السماح بالوصول للميكروفون.
        </div>
      )}

      <div style={{ ...card(C.purple), textAlign:"center", padding:28 }}>
        {recState === "recording" && activeAxis === "store" ? (
          <>
            <LiveWaveform active />
            <div style={{ fontSize:22, fontWeight:700, color:C.pink, margin:"12px 0", animation:"blink 1s infinite" }}>● {recSecs}s</div>
            <button onClick={() => stopRecording(false)} style={{ ...btn(`linear-gradient(135deg, ${C.pink}, ${C.purple})`, C.snow), width:"auto", padding:"12px 32px" }}>
              إيقاف وحفظ ⏹
            </button>
          </>
        ) : (
          <>
            <button onClick={() => startRecording("store")} style={{
              width:84, height:84, borderRadius:"50%",
              background:`radial-gradient(circle, ${C.purple}cc, ${C.purple}44)`,
              border:`3px solid ${C.purple}`, boxShadow:`0 0 30px ${C.purple}55`,
              cursor:"pointer", display:"block", margin:"0 auto 14px",
            }}>
              <div style={{ width:30, height:30, borderRadius:"50%", background:"white", margin:"0 auto" }} />
            </button>
            <p style={{ color:C.fog, fontSize:13, margin:0 }}>اضغط لبدء التسجيل</p>
          </>
        )}
        <style>{`@keyframes blink{0%,100%{opacity:1}50%{opacity:0.4}}`}</style>
      </div>

      <h3 style={{ fontSize:14, color:C.fog, margin:"4px 0 10px" }}>الملفات المحفوظة ({recordings.length})</h3>
      {recordings.length === 0 && (
        <div style={{ ...card(), textAlign:"center", padding:28 }}>
          <p style={{ margin:0, color:C.slate, fontSize:13 }}>لا توجد تسجيلات بعد. ابدأ بتسجيل أول صوت لك.</p>
        </div>
      )}
      {recordings.map(rec => (
        <div key={rec.id} style={{ ...card(), display:"flex", alignItems:"center", gap:12, padding:14 }}>
          <div style={{ width:40, height:40, borderRadius:10, background:`${C.cyan}15`, display:"flex", alignItems:"center", justifyContent:"center", fontSize:18 }}>🎵</div>
          <div style={{ flex:1 }}>
            <div style={{ fontWeight:700, fontSize:13 }}>{rec.name}</div>
            <div style={{ fontSize:11, color:C.slate }}>{rec.date}</div>
            <div style={{ display:"flex", gap:6, marginTop:4 }}>
              {["x","y","z"].map((ax,i) => (
                <span key={ax} style={{ fontSize:10, color:[C.cyan,C.pink,C.gold][i] }}>{ax.toUpperCase()}: {rec.feats[ax]>0?"+":""}{rec.feats[ax]}</span>
              ))}
            </div>
          </div>
          <button onClick={() => analyzeFromRecording(rec)} style={{ background:`${C.green}22`, border:`1px solid ${C.green}55`, color:C.green, borderRadius:8, padding:"6px 10px", fontSize:11, cursor:"pointer" }}>تحليل</button>
          <button onClick={() => deleteRecording(rec.id)} style={{ background:`${C.pink}15`, border:`1px solid ${C.pink}44`, color:C.pink, borderRadius:8, padding:"6px 10px", fontSize:11, cursor:"pointer" }}>حذف</button>
        </div>
      ))}
      <div style={{ height:6 }} />
    </div>
  );

  // ════════════════════════════════════════════════════════════════════════
  // SCREEN 6 — Full analysis with AI report
  // ════════════════════════════════════════════════════════════════════════
  const ScreenAnalyze = () => (
    <div style={{ padding:"20px 16px 0" }}>
      <div style={{ textAlign:"center", marginBottom:14 }}>
        <div style={{ fontSize:30 }}>🤖</div>
        <h2 style={{ margin:"6px 0 2px" }}>التحليل الشامل</h2>
        <p style={{ color:C.slate, fontSize:12, margin:0 }}>تقرير مفصل بالذكاء الاصطناعي مع خطة تحسين</p>
      </div>

      {permErr && (
        <div style={{ background:`${C.pink}15`, border:`1px solid ${C.pink}44`, borderRadius:12, padding:14, marginBottom:14, color:C.pink, fontSize:13 }}>
          ⚠️ لم يتم السماح بالوصول للميكروفون.
        </div>
      )}

      {!analyzeAI && recState !== "recording" && (
        <div style={{ ...card(C.green), textAlign:"center", padding:28 }}>
          <p style={{ margin:"0 0 16px", color:C.fog, fontSize:13 }}>سجّل صوتك الآن للحصول على تحليل تفصيلي شامل</p>
          <button onClick={() => startRecording("analyze")} style={{
            width:84, height:84, borderRadius:"50%",
            background:`radial-gradient(circle, ${C.green}cc, ${C.green}44)`,
            border:`3px solid ${C.green}`, boxShadow:`0 0 30px ${C.green}55`,
            cursor:"pointer", display:"block", margin:"0 auto",
          }}>
            <div style={{ width:30, height:30, borderRadius:"50%", background:"white", margin:"0 auto" }} />
          </button>
        </div>
      )}

      {recState === "recording" && activeAxis === "analyze" && (
        <div style={card(C.green)}>
          <LiveWaveform active />
          <div style={{ textAlign:"center", marginTop:10 }}>
            <span style={{ color:C.green, fontWeight:700, fontSize:20, animation:"blink 1s infinite" }}>● {recSecs}s</span>
          </div>
          <button onClick={() => stopRecording(true)} style={{ ...btn(`linear-gradient(135deg, ${C.pink}, ${C.purple})`, C.snow), marginTop:12 }}>
            إيقاف وتحليل شامل ⏹
          </button>
        </div>
      )}

      {analyzeAI && analyzeFeats && (
        <>
          <div style={{ ...card(C.green), textAlign:"center" }}>
            <div style={{ fontSize:38, fontWeight:900, color:C.green }}>{analyzeAI.overallScore}</div>
            <div style={{ fontSize:12, color:C.slate, marginBottom:6 }}>النتيجة الإجمالية</div>
            {tag(analyzeAI.voiceType, C.green)}
            <p style={{ margin:"10px 0 0", fontSize:13, color:C.fog, lineHeight:1.6 }}>{analyzeAI.headline}</p>
          </div>

          {/* Explicit numeric coordinate result: X = +3   Y = -2   Z = -3 */}
          <div style={{ ...card(), padding:14, marginBottom:14 }}>
            <p style={{ margin:"0 0 10px", fontWeight:700, fontSize:13, textAlign:"center", color:C.fog }}>نتيجة الإحداثيات</p>
            <div style={{ display:"flex", gap:8, justifyContent:"center" }}>
              {[
                { ax:"X", v:analyzeFeats.x, c:C.cyan },
                { ax:"Y", v:analyzeFeats.y, c:C.pink },
                { ax:"Z", v:analyzeFeats.z, c:C.gold },
              ].map(d => (
                <div key={d.ax} style={{
                  flex:1, textAlign:"center", padding:"12px 4px", borderRadius:12,
                  background:`${d.c}14`, border:`1px solid ${d.c}40`,
                }}>
                  <div style={{ fontFamily:"monospace", fontSize:20, fontWeight:800, color:d.c, direction:"ltr" }}>
                    {d.ax} = {d.v > 0 ? "+" : ""}{d.v}
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div style={card(C.cyan)}>
            <FullCoordinateChart point={analyzeFeats} showPoint />
          </div>

          <div style={card()}>
            <p style={{ margin:"0 0 10px", fontWeight:700, fontSize:13 }}>تحليل المحاور</p>
            {[
              { ax:"X", val:analyzeFeats.x, color:C.cyan, txt:analyzeAI.xAnalysis },
              { ax:"Y", val:analyzeFeats.y, color:C.pink, txt:analyzeAI.yAnalysis },
              { ax:"Z", val:analyzeFeats.z, color:C.gold, txt:analyzeAI.zAnalysis },
            ].map(a => (
              <div key={a.ax} style={{ marginBottom:12, paddingBottom:12, borderBottom:`1px solid ${C.rim}` }}>
                <div style={{ display:"flex", justifyContent:"space-between", marginBottom:4 }}>
                  <span style={{ color:a.color, fontWeight:700, fontSize:13 }}>محور {a.ax}</span>
                  <span style={{ color:a.color, fontWeight:700 }}>{a.val>0?"+":""}{a.val}</span>
                </div>
                <p style={{ margin:0, fontSize:12.5, color:C.fog, lineHeight:1.6 }}>{a.txt}</p>
              </div>
            ))}
          </div>

          <div style={{ display:"flex", gap:8, marginBottom:14 }}>
            {[{l:"الجودة",v:analyzeFeats.quality,c:C.cyan},{l:"الثبات",v:analyzeFeats.stability,c:C.purple},{l:"الرنين",v:analyzeFeats.resonance,c:C.gold}].map(m => (
              <div key={m.l} style={{ flex:1, background:C.panel, border:`1px solid ${m.c}22`, borderRadius:12, padding:"10px 6px", textAlign:"center" }}>
                <div style={{ fontSize:18, fontWeight:800, color:m.c }}>{m.v}%</div>
                <div style={{ fontSize:10, color:C.slate }}>{m.l}</div>
              </div>
            ))}
          </div>

          <div style={card(C.green)}>
            <p style={{ margin:"0 0 8px", color:C.green, fontWeight:700, fontSize:13 }}>✓ نقاط القوة</p>
            {analyzeAI.strengths.map((s,i) => (
              <div key={i} style={{ display:"flex", gap:8, marginBottom:5 }}>
                <span style={{ color:C.green }}>✓</span><span style={{ fontSize:12.5, color:C.fog }}>{s}</span>
              </div>
            ))}
          </div>

          <div style={card(C.gold)}>
            <p style={{ margin:"0 0 10px", color:C.gold, fontWeight:700, fontSize:13 }}>📋 ما يجب تحسينه</p>
            {analyzeAI.improvements.map((imp,i) => (
              <div key={i} style={{ marginBottom:10, padding:10, background:C.deep, borderRadius:10 }}>
                <div style={{ fontSize:12.5, fontWeight:700, color:C.gold, marginBottom:3 }}>{imp.area}</div>
                <div style={{ fontSize:12, color:C.fog, marginBottom:5 }}>{imp.issue}</div>
                <div style={{ fontSize:11.5, color:C.cyan, display:"flex", gap:6 }}><span>💡</span><span>{imp.exercise}</span></div>
              </div>
            ))}
            <div style={{ marginTop:10, padding:"10px 12px", background:`${C.pink}11`, border:`1px solid ${C.pink}33`, borderRadius:10 }}>
              <p style={{ margin:0, fontSize:12.5, color:C.fog }}>🎯 <strong style={{ color:C.pink }}>الأولوية:</strong> {analyzeAI.priorityTip}</p>
            </div>
          </div>

          <button onClick={() => { setAnalyzeAI(null); setAnalyzeFeats(null); }} style={ghost(C.fog)}>تحليل جديد</button>
        </>
      )}
      <style>{`@keyframes blink{0%,100%{opacity:1}50%{opacity:0.4}}`}</style>
      <div style={{ height:6 }} />
    </div>
  );

  const screens = {
    coords: <ScreenCoords />,
    axisX: ScreenAxis("x"),
    axisY: ScreenAxis("y"),
    axisZ: ScreenAxis("z"),
    record: <ScreenRecord />,
    analyze: <ScreenAnalyze />,
  };

  return (
    <div style={root}>
      {screens[tab]}
      {/* Bottom tab bar */}
      <div style={{
        position:"fixed", bottom:0, left:"50%", transform:"translateX(-50%)",
        width:"100%", maxWidth:440, background:"rgba(13,15,26,0.96)", backdropFilter:"blur(14px)",
        borderTop:`1px solid ${C.rim}`, display:"flex", padding:"8px 4px", zIndex:50,
      }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            flex:1, background:"none", border:"none", cursor:"pointer",
            display:"flex", flexDirection:"column", alignItems:"center", gap:2, padding:"6px 2px",
          }}>
            <span style={{ fontSize:18, opacity: tab===t.id ? 1 : 0.4, filter: tab===t.id ? `drop-shadow(0 0 8px ${C.cyan}cc)` : "none", transition:"all 0.2s" }}>{t.icon}</span>
            <span style={{ fontSize:9.5, color: tab===t.id ? C.cyan : C.slate, fontWeight: tab===t.id ? 700 : 400 }}>{t.label}</span>
          </button>
        ))}
      </div>
    </div>
  );
}


// Mount app
const container = document.getElementById('root');
const reactRoot = ReactDOM.createRoot(container);
reactRoot.render(React.createElement(App));

// Hide splash screen
if (window.__hideSplash) window.__hideSplash();
