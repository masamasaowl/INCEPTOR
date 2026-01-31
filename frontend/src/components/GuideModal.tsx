import { X, Mic, CheckCircle2, Shield, Zap } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'

interface GuideModalProps {
  isOpen: boolean
  onClose: () => void
}

const GuideModal = ({ isOpen, onClose }: GuideModalProps) => {
  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 bg-black/50 backdrop-blur-sm z-40"
          />

          {/* Modal */}
          <motion.div
            initial={{ opacity: 0, scale: 0.9, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.9, y: 20 }}
            className="fixed inset-0 z-50 flex items-center justify-center p-4"
          >
            <div className="bg-white rounded-2xl max-w-4xl w-full max-h-[90vh] overflow-y-auto shadow-2xl">
              {/* Header */}
              <div className="sticky top-0 bg-gradient-to-r from-primary-600 to-secondary-500 text-white p-6 flex items-center justify-between">
                <div>
                  <h2 className="text-2xl font-bold mb-1">🎙️ How to Use INCEPTOR</h2>
                  <p className="text-white/80">Voice Biometric Authentication Guide</p>
                </div>
                <button
                  onClick={onClose}
                  className="bg-white/20 hover:bg-white/30 p-2 rounded-full transition-all"
                >
                  <X className="w-6 h-6" />
                </button>
              </div>

              {/* Content */}
              <div className="p-6 space-y-8">
                {/* Quick Start */}
                <section>
                  <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
                    <Zap className="w-6 h-6 text-yellow-500" />
                    Quick Start (30 seconds)
                  </h3>
                  <div className="grid md:grid-cols-3 gap-4">
                    <div className="bg-gradient-to-br from-blue-50 to-blue-100 p-4 rounded-lg">
                      <div className="w-12 h-12 bg-blue-500 text-white rounded-full flex items-center justify-center font-bold text-xl mb-3">
                        1
                      </div>
                      <h4 className="font-semibold text-blue-900 mb-2">Register</h4>
                      <p className="text-sm text-blue-800">
                        Enter username, record voice 3 times
                      </p>
                    </div>
                    
                    <div className="bg-gradient-to-br from-green-50 to-green-100 p-4 rounded-lg">
                      <div className="w-12 h-12 bg-green-500 text-white rounded-full flex items-center justify-center font-bold text-xl mb-3">
                        2
                      </div>
                      <h4 className="font-semibold text-green-900 mb-2">Authenticate</h4>
                      <p className="text-sm text-green-800">
                        Enter username, record once, verify!
                      </p>
                    </div>
                    
                    <div className="bg-gradient-to-br from-purple-50 to-purple-100 p-4 rounded-lg">
                      <div className="w-12 h-12 bg-purple-500 text-white rounded-full flex items-center justify-center font-bold text-xl mb-3">
                        3
                      </div>
                      <h4 className="font-semibold text-purple-900 mb-2">Manage</h4>
                      <p className="text-sm text-purple-800">
                        View users, stats, delete profiles
                      </p>
                    </div>
                  </div>
                </section>

                {/* Registration Guide */}
                <section>
                  <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
                    <Mic className="w-6 h-6 text-primary-500" />
                    Registration Tips
                  </h3>
                  <div className="bg-blue-50 border-2 border-blue-200 rounded-lg p-4 space-y-2">
                    <p className="flex items-start gap-2 text-sm text-blue-900">
                      <CheckCircle2 className="w-5 h-5 flex-shrink-0 mt-0.5" />
                      <span><strong>Environment:</strong> Record in a quiet place with minimal background noise</span>
                    </p>
                    <p className="flex items-start gap-2 text-sm text-blue-900">
                      <CheckCircle2 className="w-5 h-5 flex-shrink-0 mt-0.5" />
                      <span><strong>Consistency:</strong> Use the same phrase: "Hello, this is my voice"</span>
                    </p>
                    <p className="flex items-start gap-2 text-sm text-blue-900">
                      <CheckCircle2 className="w-5 h-5 flex-shrink-0 mt-0.5" />
                      <span><strong>Volume:</strong> Speak at a consistent, comfortable volume across all 3 samples</span>
                    </p>
                    <p className="flex items-start gap-2 text-sm text-blue-900">
                      <CheckCircle2 className="w-5 h-5 flex-shrink-0 mt-0.5" />
                      <span><strong>Microphone:</strong> Position 6-12 inches from your mouth</span>
                    </p>
                    <p className="flex items-start gap-2 text-sm text-blue-900">
                      <CheckCircle2 className="w-5 h-5 flex-shrink-0 mt-0.5" />
                      <span><strong>Duration:</strong> Each recording is 3 seconds - pace yourself naturally</span>
                    </p>
                  </div>
                </section>

                {/* How It Works */}
                <section>
                  <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
                    <Shield className="w-6 h-6 text-primary-500" />
                    How Authentication Works
                  </h3>
                  <div className="space-y-4">
                    <div className="bg-gradient-to-r from-purple-50 to-pink-50 rounded-lg p-4">
                      <h4 className="font-semibold text-gray-800 mb-2">🔬 Feature Extraction (172 Features)</h4>
                      <p className="text-sm text-gray-700 mb-2">
                        System extracts unique voice characteristics:
                      </p>
                      <ul className="text-sm text-gray-600 space-y-1 ml-4">
                        <li>• MFCCs - Vocal tract shape (voice "fingerprint")</li>
                        <li>• Pitch - Fundamental frequency (high/low voice)</li>
                        <li>• Formants - Vocal resonances (anatomy-specific)</li>
                        <li>• Spectral features - Voice texture and timbre</li>
                      </ul>
                    </div>

                    <div className="bg-gradient-to-r from-blue-50 to-cyan-50 rounded-lg p-4">
                      <h4 className="font-semibold text-gray-800 mb-2">📊 Multi-Metric Verification (4 Algorithms)</h4>
                      <div className="grid grid-cols-2 gap-3 text-sm">
                        <div>
                          <p className="font-semibold text-blue-900">Cosine Similarity (30%)</p>
                          <p className="text-gray-600">Pattern matching</p>
                        </div>
                        <div>
                          <p className="font-semibold text-blue-900">Euclidean Distance (20%)</p>
                          <p className="text-gray-600">Point-to-point</p>
                        </div>
                        <div>
                          <p className="font-semibold text-blue-900">Bhattacharyya (35%)</p>
                          <p className="text-gray-600">Distribution comparison</p>
                        </div>
                        <div>
                          <p className="font-semibold text-blue-900">KS-Test (15%)</p>
                          <p className="text-gray-600">Statistical validation</p>
                        </div>
                      </div>
                      <p className="text-sm text-gray-600 mt-3">
                        ✅ Need <strong>3 out of 4</strong> metrics to pass for authentication
                      </p>
                    </div>

                    <div className="bg-gradient-to-r from-green-50 to-emerald-50 rounded-lg p-4">
                      <h4 className="font-semibold text-gray-800 mb-2">🎯 Adaptive Calibration</h4>
                      <p className="text-sm text-gray-700">
                        Thresholds are automatically calibrated based on YOUR voice characteristics during registration.
                        This means the system learns what's normal for you and adjusts accordingly!
                      </p>
                    </div>
                  </div>
                </section>

                {/* Troubleshooting */}
                <section>
                  <h3 className="text-xl font-bold text-gray-800 mb-4">🔧 Troubleshooting</h3>
                  <div className="space-y-3">
                    <details className="bg-gray-50 rounded-lg p-4 cursor-pointer">
                      <summary className="font-semibold text-gray-800">
                        ❌ Authentication keeps failing for me
                      </summary>
                      <div className="mt-2 text-sm text-gray-600 space-y-1">
                        <p>• Try recording in the same environment as registration</p>
                        <p>• Ensure consistent volume and distance from mic</p>
                        <p>• Check for background noise</p>
                        <p>• Consider re-registering with clearer samples</p>
                      </div>
                    </details>

                    <details className="bg-gray-50 rounded-lg p-4 cursor-pointer">
                      <summary className="font-semibold text-gray-800">
                        🎤 Microphone not working
                      </summary>
                      <div className="mt-2 text-sm text-gray-600 space-y-1">
                        <p>• Grant microphone permission when prompted</p>
                        <p>• Check browser settings (chrome://settings/content/microphone)</p>
                        <p>• Try refreshing the page</p>
                        <p>• Use HTTPS or localhost (required for microphone access)</p>
                      </div>
                    </details>

                    <details className="bg-gray-50 rounded-lg p-4 cursor-pointer">
                      <summary className="font-semibold text-gray-800">
                        ⚠️ Server connection failed
                      </summary>
                      <div className="mt-2 text-sm text-gray-600 space-y-1">
                        <p>• Make sure the backend server is running on port 8000</p>
                        <p>• Run: <code className="bg-gray-200 px-2 py-1 rounded">python3 server.py</code></p>
                        <p>• Check that CORS is enabled on the server</p>
                      </div>
                    </details>
                  </div>
                </section>

                {/* Security Info */}
                <section className="bg-gradient-to-br from-orange-50 to-red-50 border-2 border-orange-200 rounded-lg p-6">
                  <h3 className="text-xl font-bold text-orange-900 mb-4 flex items-center gap-2">
                    <Shield className="w-6 h-6" />
                    Security & Privacy
                  </h3>
                  <div className="space-y-2 text-sm text-orange-800">
                    <p>🔒 <strong>Local Storage:</strong> Voice features stored locally, not raw audio</p>
                    <p>🔐 <strong>Non-Reversible:</strong> Cannot reconstruct voice from stored features</p>
                    <p>🎯 <strong>Multi-Layer:</strong> 4 independent verification metrics</p>
                    <p>📊 <strong>Transparent:</strong> See exactly which metrics passed/failed</p>
                    <p>⚡ <strong>Real-Time:</strong> No cloud processing, all local</p>
                  </div>
                </section>

                {/* Close Button */}
                <div className="text-center">
                  <button
                    onClick={onClose}
                    className="bg-gradient-to-r from-primary-600 to-secondary-500 text-white px-8 py-3 rounded-lg font-semibold hover:shadow-lg transition-all"
                  >
                    Got it! Let's start
                  </button>
                </div>
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  )
}

export default GuideModal