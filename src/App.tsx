import React, { useState, useMemo } from 'react';
import { 
  Database, 
  Cpu, 
  Search, 
  Download, 
  FileText, 
  AlertCircle,
  RefreshCw,
  Table as TableIcon,
  Zap,
  Printer
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { GoogleGenAI } from "@google/genai";
import Papa from 'papaparse';
import ReactMarkdown from 'react-markdown';

// Initialize Gemini safely
const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY || '' });

export default function App() {
  const [data, setData] = useState<any[]>([]);
  const [headers, setHeaders] = useState<string[]>([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [report, setReport] = useState<string | null>(null);
  const [activeView, setActiveView] = useState<'table' | 'report'>('table');

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setIsProcessing(true);
    Papa.parse(file, {
      header: true,
      skipEmptyLines: true,
      complete: (results) => {
        if (results.data.length > 0) {
          setHeaders(Object.keys(results.data[0]));
          setData(results.data);
          setActiveView('table');
        }
        setIsProcessing(false);
      },
      error: () => setIsProcessing(false)
    });
  };

  const generateReport = async () => {
    if (data.length === 0) return;
    
    const API_KEY = process.env.GEMINI_API_KEY;
    if (!API_KEY || API_KEY === "") {
       setReport("### ⚠️ API Key Missing\nTo use the analysis engine in VS Code, you must:\n1. Create a `.env` file in the root folder.\n2. Add `GEMINI_API_KEY=your_key_here` to it.\n3. Restart your dev server.");
       setActiveView('report');
       return;
    }

    setIsProcessing(true);
    try {
      // Prioritize identifying top risk segments for the ML report
      const topRisks = [...filteredData]
        .sort((a, b) => (b._predicted_risk || 0) - (a._predicted_risk || 0))
        .slice(0, 10);

      const sample = JSON.stringify(topRisks);
      const prompt = `Analyze this dataset of churn risk scores produced by the CORE_LOGIC heuristic engine.
      Total Records: ${data.length}
      Top Risk Sample: ${sample}
      
      Structure the response as a clean, professional "Executive Synthesis" using these exact sections:
      
      ### 📊 Risk Architecture Analysis
      Explain why the high-risk segments are being flagged (focus on Contract patterns and Tenure stability). Use bullet points.
      
      ### 🔍 Feature Importance & Correlations
      Describe how Variables like MonthlyCharges and Contract Type are driving the scores in this dataset.
      
      ### 🛡️ Recommended Strategic Actions
      Provide 3-4 specific, high-impact technical or business interventions to reduce churn in these specific segments.
      
      ### 📈 Reliability Metrics
      A brief statement on the confidence level of this specific analysis.
      
      Technical constraints:
      - Use professional, direct language.
      - DO NOT use memorandum headers (To/From/Subject).
      - DO NOT use AI/Gemini branding.
      - Use Markdown for bolding and lists.`;

      const response = await ai.models.generateContent({
        model: "gemini-3-flash-preview",
        contents: [{
          parts: [{
            text: prompt
          }]
        }]
      });
      
      setReport(response.text ?? "No analysis generated.");
      setActiveView('report');
    } catch (error) {
      console.error("Analysis Error:", error);
      setReport("Failed to generate report. Please check system logs.");
    } finally {
      setIsProcessing(false);
    }
  };

  const filteredData = useMemo(() => {
    const rawFiltered = !searchTerm ? data : data.filter(row => 
      Object.values(row).some(val => 
        String(val).toLowerCase().includes(searchTerm.toLowerCase())
      )
    );

    // Heuristic Prediction Engine (Statistical Weighting Model)
    return rawFiltered.map(row => {
      let score = 0;
      
      // Feature 1: Contract Elasticity
      const contract = String(row.Contract || row.contract || '').toLowerCase();
      if (contract.includes('month')) score += 45;
      else if (contract.includes('one')) score += 15;
      
      // Feature 2: Tenure / Behavioral Maturity
      const tenure = Number(row.tenure || row.Tenure || 0);
      if (tenure < 6 && tenure > 0) score += 35;
      else if (tenure < 12) score += 20;
      else if (tenure > 48) score -= 25;

      // Feature 3: Revenue Variance
      const charges = Number(row.MonthlyCharges || row.monthlycharges || 0);
      if (charges > 100) score += 25;
      else if (charges > 70) score += 15;

      const probability = Math.min(Math.max(score, 4), 99);
      return { ...row, _predicted_risk: probability };
    });
  }, [data, searchTerm]);

  return (
    <div className="min-h-screen bg-[#020617] text-slate-100 font-sans selection:bg-indigo-500/30 selection:text-white flex flex-col relative overflow-hidden">
      {/* Decorative Noise / Grain Overlay */}
      <div className="absolute inset-0 pointer-events-none opacity-[0.03] z-50 bg-grain" />

      {/* Atmospheric Background Layers */}
      <div className="absolute inset-0 pointer-events-none overflow-hidden">
        <motion.div 
          animate={{ 
            scale: [1, 1.2, 1],
            x: [0, 50, 0],
            y: [0, -30, 0]
          }}
          transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
          className="absolute top-[-20%] left-[-10%] w-[60%] h-[60%] bg-indigo-600/10 rounded-full blur-[140px]" 
        />
        <motion.div 
          animate={{ 
            scale: [1, 1.1, 1],
            x: [0, -40, 0],
            y: [0, 40, 0]
          }}
          transition={{ duration: 15, repeat: Infinity, ease: "linear" }}
          className="absolute bottom-[-10%] right-[-10%] w-[50%] h-[50%] bg-blue-600/10 rounded-full blur-[140px]" 
        />
        <div className="absolute top-[30%] left-[20%] w-[400px] h-[400px] bg-indigo-500/5 rounded-full blur-[100px]" />
      </div>

      {/* Utility Header */}
      <header className="h-16 border-b border-white/10 bg-slate-900/40 backdrop-blur-md flex items-center justify-between px-6 shrink-0 sticky top-0 z-50">
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-2.5 group cursor-default">
            <div className="w-8 h-8 bg-indigo-600 rounded-lg flex items-center justify-center shadow-lg shadow-indigo-500/20 group-hover:scale-110 transition-transform">
              <Zap className="w-5 h-5 text-white fill-white" />
            </div>
            <h1 className="text-base font-black tracking-tight uppercase flex items-baseline font-display">
              ChurnLabs
              <span className="text-[10px] text-indigo-400 font-mono ml-2 opacity-70 font-sans">CORE_v2.0</span>
            </h1>
          </div>
          <div className="h-6 w-px bg-white/10 mx-2" />
          <nav className="flex gap-2">
            {[
              { id: 'table', label: 'Data Explorer', icon: TableIcon },
              { id: 'report', label: 'Analysis Insights', icon: Cpu },
            ].map((tab) => (
              <button 
                key={tab.id}
                onClick={() => setActiveView(tab.id as any)}
                className={`flex items-center gap-2 px-4 py-1.5 text-[11px] font-bold uppercase transition-all rounded-full border ${
                  activeView === tab.id 
                    ? 'bg-white/10 border-white/20 text-indigo-400 shadow-inner' 
                    : 'border-transparent text-slate-400 hover:text-white hover:bg-white/5'
                }`}
              >
                <tab.icon className="w-3.5 h-3.5" />
                {tab.label}
              </button>
            ))}
          </nav>
        </div>

        <div className="flex items-center gap-3">
          {data.length > 0 && (
             <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 bg-white/5 border border-white/10 rounded-lg text-[10px] font-mono font-bold text-slate-400">
                <Database className="w-3 h-3" />
                RECORDS: {data.length.toLocaleString()}
             </div>
          )}
          <label className="group flex items-center gap-2 px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-xs font-black uppercase tracking-wider cursor-pointer transition-all shadow-lg shadow-indigo-600/20 active:scale-95">
            <Download className="w-4 h-4 group-hover:translate-y-0.5 transition-transform" />
            Ingest Dataset
            <input type="file" accept=".csv" className="hidden" onChange={handleFileUpload} />
          </label>
        </div>
      </header>

      {/* Main Workspace */}
      <main className="flex-grow flex flex-col min-h-0 relative z-10">
        <AnimatePresence mode="wait">
          {data.length === 0 ? (
            <motion.div 
              key="empty"
              initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 1.05 }}
              className="flex-grow flex flex-col items-center justify-center p-6 text-center"
            >
               <div className="relative mb-8">
                 <div className="absolute inset-0 bg-indigo-500/20 blur-3xl rounded-full" />
                 <div className="w-20 h-20 bg-slate-800/80 backdrop-blur-xl rounded-3xl flex items-center justify-center border border-white/10 shadow-2xl relative">
                    <Database className="w-10 h-10 text-indigo-400" />
                 </div>
               </div>
                <h2 className="text-4xl font-bold text-white mb-4 tracking-tight font-display">
                  Synthesis Pipeline <span className="text-indigo-400">Ready</span>
                </h2>
                <p className="text-slate-400 max-w-lg leading-relaxed mb-12 text-sm font-medium">
                  The ChurnLabs core engine is primed for ingestion. Upload your raw customer vectors to initialize heuristic discovery.
                </p>
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6 max-w-5xl w-full">
                   {[
                     { icon: TableIcon, title: 'Fluid Explorer', desc: 'Hardware-accelerated data grid with instant vector filtering and polymorphic search support.', size: 'lg' },
                     { icon: Cpu, title: 'Heuristic Engine', desc: 'Automated pattern recognition engine for deep behavioral insights.', size: 'sm' },
                     { icon: AlertCircle, title: 'Risk Guard', desc: 'Automated retention modeling with real-time heuristic validation and strategy synthesis.', size: 'sm' },
                   ].map((feat, i) => (
                    <div key={i} className={`p-8 bg-white/[0.03] backdrop-blur-md border border-white/10 rounded-3xl text-left hover:bg-white/[0.06] transition-all group relative overflow-hidden ${feat.size === 'lg' ? 'sm:col-span-2' : ''}`}>
                       <div className="absolute top-0 right-0 w-32 h-32 bg-indigo-500/5 blur-3xl -mr-16 -mt-16 group-hover:bg-indigo-500/10 transition-colors" />
                       <div className="w-12 h-12 rounded-2xl bg-indigo-500/10 flex items-center justify-center mb-6 group-hover:scale-110 transition-all border border-indigo-500/20">
                        <feat.icon className="w-6 h-6 text-indigo-400" />
                       </div>
                       <h3 className="text-xs font-bold uppercase tracking-[0.2em] text-white/90 mb-3 font-display">{feat.title}</h3>
                       <p className="text-[13px] text-slate-400 leading-relaxed font-medium">{feat.desc}</p>
                    </div>
                  ))}
               </div>
            </motion.div>
          ) : (
            <motion.div 
              key="active" initial={{ opacity: 0 }} animate={{ opacity: 1 }}
              className="flex-grow flex flex-col min-h-0"
            >
              {activeView === 'table' ? (
                <div className="flex-grow flex flex-col min-h-0 bg-slate-900/20">
                   {/* Table Controls */}
                   <div className="px-6 py-4 bg-slate-900/40 backdrop-blur-md border-b border-white/10 flex items-center justify-between shrink-0">
                      <div className="relative w-full max-w-lg">
                        <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
                        <input 
                          type="text"
                          placeholder="Filter records by any field..."
                          value={searchTerm}
                          onChange={(e) => setSearchTerm(e.target.value)}
                          className="w-full pl-11 pr-4 py-2.5 bg-white/5 border border-white/10 rounded-xl text-[13px] text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 focus:bg-white/10 transition-all font-medium"
                        />
                      </div>
                      <button 
                        onClick={generateReport}
                        disabled={isProcessing}
                        className="ml-4 px-6 py-2.5 bg-indigo-600 text-white rounded-xl text-xs font-black uppercase tracking-widest hover:bg-indigo-500 disabled:opacity-40 transition-all flex items-center gap-2.5 shadow-lg shadow-indigo-600/20"
                      >
                        {isProcessing ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Cpu className="w-4 h-4" />}
                        Execute Analysis
                      </button>
                   </div>

                   {/* Technical Data Grid */}
                   <div className="flex-grow overflow-auto custom-scrollbar">
                      <table className="w-full border-collapse text-[12px]">
                        <thead className="sticky top-0 bg-slate-900/80 backdrop-blur-md border-b border-white/10 z-10 shadow-lg">
                          <tr>
                            <th className="px-6 py-4 text-left font-black text-indigo-400 uppercase tracking-widest border-r border-white/5 bg-indigo-500/5">
                              PREDICTED_RISK
                            </th>
                            {headers.map(h => (
                              <th key={h} className="px-6 py-4 text-left font-black text-slate-400 uppercase tracking-widest border-r border-white/5">
                                {h.replace(/_/g, ' ')}
                              </th>
                            ))}
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-white/5">
                          {filteredData.slice(0, 100).map((row, i) => (
                            <tr key={i} className="hover:bg-indigo-500/5 transition-colors group/row border-b border-white/[0.02] last:border-0">
                               <td className="px-6 py-4 border-r border-white/5 font-mono bg-white/[0.02] sticky left-0 z-10">
                                  <div className="flex items-center gap-3">
                                    <div className="w-12 h-1.5 bg-white/10 rounded-full overflow-hidden">
                                      <div 
                                        className={`h-full transition-all duration-1000 ${
                                          row._predicted_risk > 70 ? 'bg-rose-500' : 
                                          row._predicted_risk > 40 ? 'bg-amber-500' : 'bg-emerald-500'
                                        }`}
                                        style={{ width: `${row._predicted_risk}%` }}
                                      />
                                    </div>
                                    <span className={`text-[10px] font-black ${
                                       row._predicted_risk > 70 ? 'text-rose-400' : 
                                       row._predicted_risk > 40 ? 'text-amber-400' : 'text-emerald-400'
                                    }`}>
                                      {row._predicted_risk}%
                                    </span>
                                  </div>
                               </td>
                              {headers.map(h => {
                                const val = String(row[h]);
                                const isRisk = val.toLowerCase().includes('churn') || val.toLowerCase().includes('high');
                                const isStable = val.toLowerCase().includes('loyal') || val.toLowerCase().includes('low');
                                
                                return (
                                  <td key={h} className="px-6 py-4 border-r border-white/5 font-mono text-slate-400 group-hover/row:text-indigo-200 transition-colors">
                                    <span className={`px-1.5 py-0.5 rounded ${
                                      isRisk ? 'bg-rose-500/10 text-rose-400 border border-rose-500/20' : 
                                      isStable ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20' : 
                                      !isNaN(Number(val)) ? 'text-indigo-300' : ''
                                    }`}>
                                      {val}
                                    </span>
                                  </td>
                                );
                              })}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                      {filteredData.length > 100 && (
                        <div className="p-10 text-center text-slate-500 text-[11px] uppercase tracking-[0.3em] font-black bg-slate-900/40 border-t border-white/10">
                          — Partial View Enabled: {filteredData.length} total rows —
                        </div>
                      )}
                   </div>
                </div>
              ) : (
                <div className="flex-grow overflow-y-auto p-8 sm:p-12 bg-slate-900/10">
                  <div className="max-w-5xl mx-auto">
                    <div className="bg-slate-900/60 backdrop-blur-2xl border border-white/10 rounded-3xl shadow-2xl overflow-hidden">
                      <header className="px-10 py-12 border-b border-white/10 flex justify-between items-start bg-gradient-to-br from-white/5 to-transparent">
                         <div className="space-y-4">
                           <div className="inline-flex items-center gap-2 px-3 py-1 bg-indigo-500/10 border border-indigo-500/20 rounded-full">
                            <span className="w-1.5 h-1.5 rounded-full bg-indigo-400 animate-pulse" />
                            <span className="text-[10px] font-black text-indigo-300 uppercase tracking-widest">Protocol: CHURN_INTEL_v3</span>
                           </div>
                           <h2 className="text-4xl font-bold tracking-tight text-white italic font-display">RETENTION_STRATEGY_REPORT</h2>
                           <p className="text-slate-400 text-xs font-medium tracking-wide">Analysis generated based on active heuristic parameters and current dataset ingest.</p>
                         </div>
                         <div className="text-right space-y-2">
                            <p className="text-[10px] font-bold text-slate-500 uppercase tracking-widest opacity-60">Logic Engine</p>
                            <p className="text-lg font-mono font-black text-indigo-400 tracking-tighter">CORE_SYNTH_v3</p>
                            <div className="h-1 w-full bg-indigo-500/20 rounded-full overflow-hidden">
                              <motion.div initial={{ x: '-100%' }} animate={{ x: '100%' }} transition={{ duration: 2, repeat: Infinity, ease: 'linear' }} className="w-1/2 h-full bg-indigo-500" />
                            </div>
                         </div>
                      </header>

                      {!report && !isProcessing ? (
                         <div className="p-24 text-center">
                            <div className="w-20 h-20 bg-white/5 rounded-full flex items-center justify-center mx-auto mb-6 border border-white/10">
                              <FileText className="w-8 h-8 text-slate-500" />
                            </div>
                            <p className="text-sm font-bold text-slate-500 uppercase tracking-[0.3em] mb-8">Synthesis Not Initialized</p>
                            <button 
                              onClick={generateReport}
                              className="px-10 py-4 bg-white text-slate-950 rounded-xl font-black text-xs uppercase tracking-[0.3em] hover:bg-indigo-400 hover:text-white transition-all shadow-xl shadow-white/5 active:scale-95"
                            >
                              Boot_Synthesis
                            </button>
                         </div>
                      ) : isProcessing ? (
                         <div className="py-32 text-center space-y-12">
                            <div className="flex justify-center gap-3">
                               {[0, 1, 2, 3, 4].map(i => (
                                 <motion.div 
                                   key={i}
                                   animate={{ height: [24, 64, 24], opacity: [0.3, 1, 0.3], backgroundColor: ['#6366f1', '#a855f7', '#6366f1'] }}
                                   transition={{ duration: 1.2, repeat: Infinity, delay: i * 0.15, ease: 'easeInOut' }}
                                   className="w-1.5 h-6 rounded-full"
                                 />
                               ))}
                            </div>
                            <div className="space-y-3">
                              <p className="text-[11px] font-mono font-black text-white uppercase tracking-[0.6em] animate-pulse">Scanning_Customer_Vectors</p>
                              <p className="text-[9px] text-slate-500 font-mono tracking-widest">LAYER_PROCESSING_ACTIVE [{Math.floor(Math.random()*100)}%]</p>
                            </div>
                         </div>
                      ) : (
                        <motion.div 
                          initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                          className="p-10 sm:p-12 pb-20"
                        >
                           <div className="space-y-6">
                             <div className="bg-white/5 border border-white/10 rounded-2xl p-8 sm:p-12 font-sans text-[15px] text-slate-300 shadow-inner">
                               <div className="markdown-body">
                                 <ReactMarkdown>{report}</ReactMarkdown>
                               </div>
                             </div>
                           </div>
                           <div className="mt-12 flex items-center justify-between border-t border-white/10 pt-8">
                             <div className="text-[9px] font-mono text-slate-600 uppercase tracking-widest">END_OF_ARTIFACT // STATUS_VERIFIED</div>
                             <button 
                                onClick={() => window.print()}
                                className="px-6 py-2.5 bg-white/5 border border-white/10 text-[10px] font-black uppercase tracking-[0.2em] text-slate-400 hover:text-white hover:border-white/30 transition-all rounded-lg flex items-center gap-2.5"
                              >
                                <Printer className="w-3.5 h-3.5" />
                                Export_Session
                              </button>
                           </div>
                        </motion.div>
                      )}
                    </div>
                  </div>
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      {/* Technical Status Strip */}
      <footer className="h-10 px-6 border-t border-white/10 bg-slate-900/60 backdrop-blur-md flex items-center justify-between shrink-0 text-[10px] font-mono font-bold text-slate-500 select-none z-50">
        <div className="flex gap-8 items-center">
          <div className="flex items-center gap-2">
             <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 shadow-[0_0_12px_#10b981]" />
             SYSTEM: ONLINE
          </div>
          <div className="text-white/10">/</div>
          <div>PROTOCOL: SECURE_INGEST</div>
          <div className="text-white/10">/</div>
          <div>REGION: GLOBAL_EDGE</div>
        </div>
        <div className="flex gap-8 items-center">
          <div className="uppercase hidden sm:block">LOCAL_TIME: {new Date().toLocaleTimeString([], { hour12: false })}</div>
          <div className="text-white/10">/</div>
          <div className="text-indigo-400 uppercase tracking-tighter">Churn Prediction Engine</div>
        </div>
      </footer>

      {/* Custom Styles */}
      <style>{`
        .custom-scrollbar::-webkit-scrollbar {
          width: 6px;
          height: 6px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: rgba(255, 255, 255, 0.02);
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: rgba(255, 255, 255, 0.1);
          border-radius: 10px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: rgba(255, 255, 255, 0.2);
        }
      `}</style>
    </div>
  );
}
