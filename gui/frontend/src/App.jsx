import { useState, useRef, useCallback, useEffect } from 'react'
import './App.css'
import Folder from './Folder'
import DotGrid from './DotGrid'
import BorderGlow from './BorderGlow'

const API = 'https://brianlov-guibackend.hf.space'

const MODELS = ['Baseline', 'GAN', 'EBM']

const MODEL_META = {
  GAN: { color: '#4f8ef7', desc: 'ConvTranspose2d Generator + Spectral Norm Discriminator' },
  EBM: { color: '#a855f7', desc: 'Energy-Based Model via Langevin Dynamics (MCMC sampling)' },
}

// ── Upload Zone ───────────────────────────────────────────────────────────────
function UploadZone({ onUpload, preview }) {
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef()

  const handle = useCallback((file) => {
    if (file && file.type.startsWith('image/')) onUpload(file)
  }, [onUpload])

  return (
    <div
      id="upload-zone"
      className={`upload-zone ${dragging ? 'dragging' : ''} ${preview ? 'has-preview' : ''}`}
      onDragOver={e => { e.preventDefault(); setDragging(true) }}
      onDragLeave={() => setDragging(false)}
      onDrop={async e => {
        e.preventDefault();
        setDragging(false);
        const file = e.dataTransfer.files[0];
        if (file) {
          handle(file);
        } else {
          let url = e.dataTransfer.getData('URL') || e.dataTransfer.getData('text/plain');
          if (url) {
            try {
              const res = await fetch(url);
              const blob = await res.blob();
              const f = new File([blob], url.split('/').pop() || 'sample.jpg', { type: blob.type });
              handle(f);
            } catch (err) { console.error('Failed to fetch dropped image', err); }
          }
        }
      }}
      onClick={() => inputRef.current.click()}
    >
      <input ref={inputRef} type="file" accept="image/*" hidden
        onChange={e => handle(e.target.files[0])} />
      {preview ? (
        <div className="preview-wrap">
          <img src={preview} alt="Uploaded X-ray" className="preview-img" />
          <span className="preview-change">Click to change</span>
        </div>
      ) : (
        <>
          <div className="upload-icon">🫁</div>
          <p className="upload-title">Drop your chest X-ray here</p>
          <p className="upload-sub">or click to browse &mdash; PNG, JPG, JPEG</p>
        </>
      )}
    </div>
  )
}

// ── Augmentation Grid ─────────────────────────────────────────────────────────
function AugGrid({ items, loading }) {
  if (loading) return <div className="spinner-wrap"><div className="spinner" /></div>
  if (!items.length) return <p className="empty">Upload an image to see augmented versions.</p>
  return (
    <div className="img-grid">
      {items.map((item, i) => (
        <div key={i} className="img-card">
          <img src={`data:image/png;base64,${item.image}`} alt={item.label} />
          <span className="img-label">{item.label}</span>
        </div>
      ))}
    </div>
  )
}

// ── Model Sample Column ───────────────────────────────────────────────────────
function ModelColumn({ name, images }) {
  const meta = MODEL_META[name]
  return (
    <div className="model-col">
      <div className="model-col-header" style={{ borderColor: meta.color }}>
        <span className="model-name" style={{ color: meta.color }}>{name}</span>
        <span className="model-desc">{meta.desc}</span>
      </div>
      <div className="model-img-stack">
        {images.length === 0
          ? [0].map(i => <div key={i} className="img-placeholder" />)
          : images.map((img, i) => (
            <div key={i} className="img-card">
              <img src={`data:image/png;base64,${img}`} alt={`${name} sample`} />
              <span className="img-label">Sample</span>
            </div>
          ))
        }
      </div>
    </div>
  )
}

// ── Confidence Bar ────────────────────────────────────────────────────────────
function ConfidenceBar({ label, value, color }) {
  return (
    <div className="conf-row">
      <span className="conf-label">{label}</span>
      <div className="conf-track">
        <div className="conf-fill" style={{ width: `${value * 100}%`, background: color }} />
      </div>
      <span className="conf-pct">{(value * 100).toFixed(1)}%</span>
    </div>
  )
}

// ── Dashboard Component ───────────────────────────────────────────────────────
function Dashboard({ navigate }) {
  const [metrics, setMetrics] = useState(null)
  const [samples, setSamples] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([
      fetch(`${API}/metrics_images`).then(r => r.json()),
      fetch(`${API}/dataset_samples`).then(r => r.json())
    ])
      .then(([metricsData, samplesData]) => {
        setMetrics(metricsData)
        setSamples(samplesData)
        setLoading(false)
      })
      .catch(() => setLoading(false))
  }, [])

  return (
    <div className="app">
      <BorderGlow 
        className="hero" 
        backgroundColor="#060b17" 
        glowColor="220 80 60" 
        edgeSensitivity={50}
        animated={true}
        borderRadius={24}
        style={{ margin: '24px 24px 0 24px' }}
      >
        <div className="hero-content">
          <a href="/" onClick={(e) => navigate('/', e)} className="nav-link" style={{ marginBottom: '16px', display: 'inline-block' }}>
            &larr; Back to Generator
          </a>
          <h1>Model Evaluation <span className="gradient-text">Metrics</span></h1>
          <p className="hero-sub">Confusion Matrices and ROC Curves for Baseline, GAN, and EBM</p>
        </div>
      </BorderGlow>

      <main className="main">
        {loading ? (
          <div className="spinner-wrap"><div className="spinner" /><p className="spinner-label">Loading metrics and dataset samples...</p></div>
        ) : !metrics ? (
          <p className="empty">Failed to load metrics. Check backend.</p>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>

            {/* Dataset Samples Section */}
            {samples && (
              <section className="card">
                <div className="card-header">
                  <h2>PneumoniaMNIST Dataset Samples</h2>
                </div>
                <p className="card-desc">28*28 pixels binary images samples in MedMnist </p>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px', padding: '16px 0' }}>
                  <div className="model-col">
                    <div className="model-col-header" style={{ borderColor: '#10b981' }}>
                      <span className="model-name" style={{ color: '#10b981' }}>Normal (Healthy)</span>
                      <span className="model-desc">The minority class in training (Needs augmentation)</span>
                    </div>
                    <div className="model-img-stack" style={{ flexDirection: 'row', justifyContent: 'center' }}>
                      {samples.normal.map((img, i) => (
                        <img key={i} src={`data:image/png;base64,${img}`} style={{ width: '30%', borderRadius: '4px' }} alt="Normal" />
                      ))}
                    </div>
                  </div>
                  <div className="model-col">
                    <div className="model-col-header" style={{ borderColor: '#f87171' }}>
                      <span className="model-name" style={{ color: '#f87171' }}>Pneumonia</span>
                      <span className="model-desc">The majority class (Notice the cloudy opacities)</span>
                    </div>
                    <div className="model-img-stack" style={{ flexDirection: 'row', justifyContent: 'center' }}>
                      {samples.pneumonia.map((img, i) => (
                        <img key={i} src={`data:image/png;base64,${img}`} style={{ width: '30%', borderRadius: '4px' }} alt="Pneumonia" />
                      ))}
                    </div>
                  </div>
                </div>
              </section>
            )}

            {/* Explanation Section */}
            <section className="card">
              <div className="card-header">
                <h2>Why did EBM outperform GAN?</h2>
              </div>
              <div style={{ padding: '16px', fontSize: '14px', color: 'var(--text)', lineHeight: '1.6' }}>
                <p style={{ marginBottom: '12px' }}><strong>1. No Mode Collapse:</strong> GANs often suffer from mode collapse, where the generator finds a few realistic examples and repeats them. This creates sharp but repetitive images, failing to capture the full natural diversity of human lungs.</p>
                <p style={{ marginBottom: '12px' }}><strong>2. Diverse Distribution Modeling:</strong> EBMs (via Langevin dynamics) explore the entire continuous space of the data distribution. Because they assign low "energy" to all realistic lung states, EBMs generate a massive, diverse variety of healthy lungs including subtle variations in rib cage shapes and tissue densities.</p>
                <p><strong>Conclusion:</strong> When the EBM-generated "Normal" images were added to the dataset, they provided the classifier with a much wider, richer spectrum of healthy lungs to learn from. This led to a more robust decision boundary, drastically reducing <strong>False Positives</strong> from 10 (Baseline) to 5 (EBM), pushing Specificity to 96.30%.</p>
              </div>
            </section>

            {/* Metrics Section */}
            <section>
              <h2 style={{ fontSize: '18px', marginBottom: '16px' }}>Model Metrics</h2>
              <div className="metrics-grid">
                {['Baseline', 'GAN', 'EBM'].map(model => (
                  <section key={model} className="card">
                    <div className="card-header">
                      <h2 style={{ color: MODEL_META[model]?.color || 'var(--text)' }}>{model} Metrics</h2>
                    </div>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px', padding: '16px' }}>
                      <div>
                        <h3 style={{ fontSize: '12px', textTransform: 'uppercase', color: 'var(--muted)', marginBottom: '8px', textAlign: 'center' }}>Confusion Matrix</h3>
                        {metrics[model]?.cm ? (
                          <img src={`data:image/png;base64,${metrics[model].cm}`} alt={`${model} CM`} style={{ width: '100%', borderRadius: '8px', border: '1px solid var(--border)' }} />
                        ) : <div className="img-placeholder" />}
                      </div>
                      <div>
                        <h3 style={{ fontSize: '12px', textTransform: 'uppercase', color: 'var(--muted)', marginBottom: '8px', textAlign: 'center' }}>ROC Curve</h3>
                        {metrics[model]?.roc ? (
                          <img src={`data:image/png;base64,${metrics[model].roc}`} alt={`${model} ROC`} style={{ width: '100%', borderRadius: '8px', border: '1px solid var(--border)' }} />
                        ) : <div className="img-placeholder" />}
                      </div>
                    </div>
                  </section>
                ))}
              </div>
            </section>
          </div>
        )}
      </main>
    </div>
  )
}

// ── Main App ──────────────────────────────────────────────────────────────────
export default function App() {
  const [currentRoute, setCurrentRoute] = useState(window.location.pathname)
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState('')
  const [augments, setAugments] = useState([])
  const [genData, setGenData] = useState({})
  const [result, setResult] = useState(null)
  const [model, setModel] = useState('Baseline')

  const [loadingAug, setLoadingAug] = useState(false)
  const [loadingGen, setLoadingGen] = useState(false)
  const [loadingClf, setLoadingClf] = useState(false)

  useEffect(() => {
    const handleLocationChange = () => setCurrentRoute(window.location.pathname)
    window.addEventListener('popstate', handleLocationChange)
    return () => window.removeEventListener('popstate', handleLocationChange)
  }, [])

  const navigate = (path, e) => {
    if (e) e.preventDefault()
    window.history.pushState({}, '', path)
    setCurrentRoute(path)
  }

  const handleUpload = async (f) => {
    setFile(f)
    setPreview(URL.createObjectURL(f))
    setAugments([]); setGenData({}); setResult(null)

    setLoadingAug(true)
    setLoadingGen(true)
    try {
      const fd = new FormData(); fd.append('file', f)
      const [augRes, genRes] = await Promise.all([
        fetch(`${API}/augment`, { method: 'POST', body: fd }),
        fetch(`${API}/generate?n=1`, { method: 'POST' })
      ])
      const [augData, genData] = await Promise.all([augRes.json(), genRes.json()])
      setAugments(augData.augmentations)
      setGenData(genData)
    } catch { alert('Backend not reachable. Make sure FastAPI is running on port 8000.') }
    finally { setLoadingAug(false); setLoadingGen(false) }
  }

  const handleClassify = async () => {
    if (!file) return
    setLoadingClf(true); setResult(null)
    try {
      const fd = new FormData(); fd.append('file', file); fd.append('model_name', model)
      const res = await fetch(`${API}/classify`, { method: 'POST', body: fd })
      setResult(await res.json())
    } catch { alert('Classification failed. Check backend.') }
    finally { setLoadingClf(false) }
  }

  if (currentRoute === '/dashboard') {
    return <Dashboard navigate={navigate} />
  }

  return (
    <div className="app">
      <BorderGlow 
        className="hero" 
        backgroundColor="#060b17" 
        glowColor="220 80 60" 
        edgeSensitivity={50}
        animated={true}
        borderRadius={24}
        style={{ margin: '24px 24px 0 24px' }}
      >
        <div className="hero-content">
          <div className="hero-badge">CVPR 2026 &mdash; BERR 4743</div>
          <h1>Generative Medical Imaging<br /><span className="gradient-text">Augmentation Dashboard</span></h1>
          <p className="hero-sub">
            PneumoniaMNIST &bull; GAN &bull; EBM
          </p>
          <div style={{ marginTop: '16px' }}>
            <a href="/dashboard" onClick={(e) => navigate('/dashboard', e)} className="nav-link">
              View Model Metrics &rarr;
            </a>
          </div>
        </div>
      </BorderGlow>

      <main className="main">

        <BorderGlow className="card" backgroundColor="#0d1526" animated={false}>
          <div className="card-header">
            <h2>Upload Chest X-ray</h2>
            <span className="badge">Step 1</span>
          </div>
          <UploadZone onUpload={handleUpload} preview={preview} />

          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', paddingTop: '24px', marginTop: '24px', borderTop: '1px solid var(--border)' }}>
            <div>
              <h2 style={{ marginBottom: '8px', fontSize: '18px' }}>Try to Test!</h2>
              <p className="card-desc" style={{ margin: 0 }}>No X-ray? Click the folder to open it, then drag an image into the drop zone.</p>
            </div>
            <div style={{ height: '120px', width: '160px', position: 'relative', display: 'flex', justifyContent: 'center', alignItems: 'center', paddingTop: '40px' }}>
              <Folder
                size={1.2}
                color="#4785ff"
                items={[
                  <div key="1" className="sample-wrapper" title="Drag me! I am a pneumonia lung!!">
                    <img src="/sample-1.jpg" draggable="true" alt="Pneumonia Sample" />
                    <div className="cloud-tooltip">Drag me! I am a pneumonia lung!!</div>
                  </div>,
                  <div key="2" className="sample-wrapper" title="Drag me! I am a normal lung!!">
                    <img src="/sample-2.jpg" draggable="true" alt="Normal Sample" />
                    <div className="cloud-tooltip">Drag me! I am a normal lung!!</div>
                  </div>
                ]}
              />
            </div>
          </div>
        </BorderGlow>

        {/* Traditional Augmentations */}
        <BorderGlow className="card" backgroundColor="#0d1526" animated={false}>
          <div className="card-header">
            <h2>Traditional Augmentations</h2>
            <span className="badge">Row 1</span>
          </div>
          <p className="card-desc">Standard image transforms applied to the uploaded X-ray — flips, rotations, brightness, blur, and cropping.</p>
          <AugGrid items={augments} loading={loadingAug} />
        </BorderGlow>

        {/* Generative Samples — GAN & EBM */}
        <BorderGlow className="card" backgroundColor="#0d1526" animated={false}>
          <div className="card-header">
            <h2>Generative Model Samples</h2>
            <span className="badge gen-badge">Row 2 — GAN & EBM</span>
          </div>
          <p className="card-desc">
            Synthetic <strong>Normal</strong> lung images auto-generated when you upload — these are what each model added to the training set to fix class imbalance.
          </p>

          {loadingGen && (
            <div className="spinner-wrap">
              <div className="spinner" />
              <p className="spinner-label">Generating GAN &amp; EBM samples…</p>
            </div>
          )}

          {!loadingGen && Object.keys(genData).length === 0 && (
            <p className="empty">Upload an image above to auto-generate GAN & EBM samples.</p>
          )}

          {Object.keys(genData).length > 0 && (
            <div className="gen-all-grid" style={{ gridTemplateColumns: '1fr 1fr' }}>
              <ModelColumn name="GAN" images={genData.gan || []} />
              <ModelColumn name="EBM" images={genData.ebm || []} />
            </div>
          )}
        </BorderGlow>

        {/* Classification */}
        <BorderGlow className="card" backgroundColor="#0d1526" animated={false}>
          <div className="card-header">
            <h2>Classification</h2>
            <span className="badge">Step 2 — Run Inference</span>
          </div>
          <p className="card-desc">
            Select a ResNet50 model trained with each augmentation strategy and classify your uploaded X-ray.
          </p>
          <div className="clf-controls">
            <div className="select-wrap">
              <select id="model-select" value={model} onChange={e => setModel(e.target.value)}>
                {MODELS.map(m => <option key={m} value={m}>{m}</option>)}
              </select>
            </div>
            <button id="btn-classify" className="btn-primary" onClick={handleClassify}
              disabled={!file || loadingClf}>
              {loadingClf ? 'Classifying…' : '🔬 Classify'}
            </button>
          </div>

          {loadingClf && <div className="spinner-wrap"><div className="spinner" /></div>}

          {result && (
            <div className={`result-box ${result.prediction === 'Normal' ? 'normal' : 'pneumonia'}`}>
              <div className="result-header">
                <span className="result-icon">{result.prediction === 'Normal' ? '✅' : '⚠️'}</span>
                <div>
                  <div className="result-label">Prediction</div>
                  <div className="result-pred">{result.prediction}</div>
                </div>
                <div className="result-model-badge">{result.model}</div>
              </div>
              <div className="conf-bars">
                <ConfidenceBar label="Normal" value={result.prob_normal} color="var(--green)" />
                <ConfidenceBar label="Pneumonia" value={result.prob_pneumonia} color="var(--red)" />
              </div>
            </div>
          )}
        </BorderGlow>

        {/* Results Summary Table */}
        <BorderGlow className="card" backgroundColor="#0d1526" animated={false}>
          <div className="card-header">
            <h2>Model Performance Summary</h2>
            <span className="badge">PneumoniaMNIST Test Set</span>
          </div>
          <div className="table-wrap">
            <table className="results-table">
              <thead>
                <tr>
                  <th>Model</th><th>Accuracy</th><th>Sensitivity</th><th>Specificity</th><th>AUC</th><th>FP ↓</th><th>FN ↓</th>
                </tr>
              </thead>
              <tbody>
                {[
                  { name: 'Baseline', acc: 97.90, sens: 99.74, spec: 92.59, auc: 0.9983, fp: 10, fn: 1 },
                  { name: 'GAN', acc: 98.47, sens: 99.74, spec: 94.81, auc: 0.9979, fp: 7, fn: 1 },
                  { name: 'EBM', acc: 98.85, sens: 99.74, spec: 96.30, auc: 0.9993, fp: 5, fn: 1, best: true },
                ].map(r => (
                  <tr key={r.name} className={r.best ? 'row-best' : ''}>
                    <td>
                      <span className="table-model-name">
                        {r.name !== 'Baseline' && (
                          <span className="table-dot" style={{ background: MODEL_META[r.name]?.color }} />
                        )}
                        <strong>{r.name}</strong>
                        {r.best && <span className="trophy">🏆 Best</span>}
                      </span>
                    </td>
                    <td className={r.best ? 'cell-best' : ''}><strong>{r.acc.toFixed(2)}%</strong></td>
                    <td>{r.sens.toFixed(2)}%</td>
                    <td>{r.spec.toFixed(2)}%</td>
                    <td>{r.auc.toFixed(4)}</td>
                    <td className={r.fp <= 3 ? 'cell-fp-low' : r.fp >= 10 ? 'cell-worst' : ''}>{r.fp}</td>
                    <td className={r.fn <= 1 ? 'cell-fp-low' : r.fn >= 4 ? 'cell-worst' : ''}>{r.fn}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </BorderGlow>

      </main>

      <footer className="footer">
        <p>CVPR Assignment &mdash; Generative Medical Imaging Augmentation &mdash; PneumoniaMNIST</p>
      </footer>
    </div>
  )
}
