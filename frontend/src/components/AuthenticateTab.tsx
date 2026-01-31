import { useState, useRef } from 'react'
import toast from 'react-hot-toast'
import { Mic, StopCircle, Loader2, CheckCircle2, XCircle, AlertCircle } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import { authenticateUser } from '../api/voiceAuth'
import MetricsDisplay from './MetricsDisplay'

const AuthenticateTab = () => {
  const [username, setUsername] = useState('')
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null)
  const [isRecording, setIsRecording] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [result, setResult] = useState<any>(null)
  
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const audioChunksRef = useRef<Blob[]>([])

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          sampleRate: 16000
        } 
      })
      
      mediaRecorderRef.current = new MediaRecorder(stream)
      audioChunksRef.current = []

      mediaRecorderRef.current.ondataavailable = (event) => {
        audioChunksRef.current.push(event.data)
      }

      mediaRecorderRef.current.onstop = () => {
        const blob = new Blob(audioChunksRef.current, { type: 'audio/wav' })
        setAudioBlob(blob)
        toast.success('Recording complete!')
        stream.getTracks().forEach(track => track.stop())
      }

      mediaRecorderRef.current.start()
      setIsRecording(true)

      // Auto-stop after 3 seconds
      setTimeout(() => {
        if (mediaRecorderRef.current && isRecording) {
          stopRecording()
        }
      }, 3000)
      
      toast.success('Recording started! Say your passphrase...')
    } catch (error) {
      toast.error('Microphone access denied')
      console.error('Microphone error:', error)
    }
  }

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop()
      setIsRecording(false)
    }
  }

  const handleAuthenticate = async () => {
    if (!username.trim()) {
      toast.error('Please enter a username')
      return
    }

    if (!audioBlob) {
      toast.error('Please record your voice first')
      return
    }

    setIsLoading(true)
    const loadingToast = toast.loading('Authenticating...')

    try {
      const data = await authenticateUser(username, audioBlob)
      
      if (data.success) {
        toast.success(`Welcome back, ${data.username}! 🎉`, { id: loadingToast })
      } else {
        toast.error('Authentication failed!', { id: loadingToast })
      }
      
      setResult(data)
    } catch (error: any) {
      toast.error(error.message || 'Authentication failed', { id: loadingToast })
      console.error('Authentication error:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const reset = () => {
    setUsername('')
    setAudioBlob(null)
    setResult(null)
  }

  return (
    <div className="space-y-6">
      {/* Instructions */}
      <div className="bg-gradient-to-r from-blue-50 to-purple-50 border border-blue-200 rounded-lg p-4">
        <h3 className="font-semibold text-blue-900 mb-2 flex items-center gap-2">
          <AlertCircle className="w-5 h-5" />
          Authentication Guide
        </h3>
        <ul className="text-sm text-blue-800 space-y-1">
          <li>• Enter your registered username</li>
          <li>• Record your voice (same phrase as registration)</li>
          <li>• System will verify using 4 different metrics</li>
          <li>• Need 3 out of 4 metrics to pass for authentication</li>
        </ul>
      </div>

      {/* Username Input */}
      <div>
        <label className="block text-sm font-semibold text-gray-700 mb-2">
          Username
        </label>
        <input
          type="text"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          placeholder="Enter your username"
          className="w-full px-4 py-3 border-2 border-gray-300 rounded-lg focus:border-primary-500 focus:ring-2 focus:ring-primary-200 outline-none transition-all"
          disabled={isLoading}
        />
      </div>

      {/* Recording Section */}
      <div className="bg-gray-50 rounded-lg p-6">
        <div className="flex justify-center mb-4">
          <button
            onClick={isRecording ? stopRecording : startRecording}
            disabled={isLoading}
            className={`w-32 h-32 rounded-full flex flex-col items-center justify-center font-bold text-white shadow-xl transition-all duration-300 ${
              isRecording
                ? 'bg-red-500 animate-pulse scale-110'
                : audioBlob
                ? 'bg-green-500'
                : 'bg-gradient-to-br from-primary-500 to-secondary-500 hover:scale-110'
            }`}
          >
            {isRecording ? (
              <>
                <StopCircle className="w-12 h-12 mb-2" />
                <span className="text-xs">Stop</span>
              </>
            ) : audioBlob ? (
              <>
                <CheckCircle2 className="w-12 h-12 mb-2" />
                <span className="text-xs">Recorded</span>
              </>
            ) : (
              <>
                <Mic className="w-12 h-12 mb-2" />
                <span className="text-xs">Record</span>
              </>
            )}
          </button>
        </div>

        {audioBlob && (
          <div className="text-center">
            <p className="text-sm text-green-600 font-semibold">✅ Voice sample ready</p>
            <button
              onClick={() => setAudioBlob(null)}
              className="text-sm text-blue-600 hover:underline mt-1"
            >
              Record again
            </button>
          </div>
        )}
      </div>

      {/* Action Buttons */}
      <div className="flex gap-4">
        <button
          onClick={reset}
          className="flex-1 bg-gray-200 text-gray-700 py-3 rounded-lg font-semibold hover:bg-gray-300 transition-all"
          disabled={isLoading}
        >
          Reset
        </button>
        <button
          onClick={handleAuthenticate}
          disabled={!username.trim() || !audioBlob || isLoading}
          className="flex-1 bg-primary-600 text-white py-3 rounded-lg font-semibold hover:bg-primary-700 transition-all duration-300 hover:shadow-lg disabled:bg-gray-400 disabled:cursor-not-allowed flex items-center justify-center gap-2"
        >
          {isLoading ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              Authenticating...
            </>
          ) : (
            'Authenticate'
          )}
        </button>
      </div>

      {/* Results */}
      <AnimatePresence>
        {result && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="space-y-4"
          >
            {/* Success/Failure Banner */}
            <div className={`border-2 rounded-lg p-6 text-center ${
              result.success
                ? 'bg-green-50 border-green-500'
                : 'bg-red-50 border-red-500'
            }`}>
              {result.success ? (
                <>
                  <CheckCircle2 className="w-16 h-16 text-green-500 mx-auto mb-4" />
                  <h3 className="text-2xl font-bold text-green-900 mb-2">
                    Authentication Successful! 🎉
                  </h3>
                  <p className="text-green-700">
                    Welcome back, <span className="font-bold">{result.username}</span>
                  </p>
                </>
              ) : (
                <>
                  <XCircle className="w-16 h-16 text-red-500 mx-auto mb-4" />
                  <h3 className="text-2xl font-bold text-red-900 mb-2">
                    Authentication Failed
                  </h3>
                  <p className="text-red-700">
                    Voice does not match registered profile
                  </p>
                </>
              )}
            </div>

            {/* Metrics Summary */}
            <div className="bg-gray-50 rounded-lg p-4">
              <div className="grid grid-cols-2 gap-4 mb-4">
                <div className="bg-white rounded-lg p-3 border border-gray-200">
                  <p className="text-sm text-gray-600">Checks Passed</p>
                  <p className="text-2xl font-bold text-primary-600">
                    {result.checks_passed}/{result.required_checks}
                  </p>
                </div>
                <div className="bg-white rounded-lg p-3 border border-gray-200">
                  <p className="text-sm text-gray-600">Combined Score</p>
                  <p className="text-2xl font-bold text-primary-600">
                    {(result.combined_score * 100).toFixed(1)}%
                  </p>
                </div>
              </div>
            </div>

            {/* Detailed Metrics */}
            <MetricsDisplay 
              metrics={result.scores} 
              thresholds={result.thresholds}
              checks={result.checks}
            />

            {/* Helpful Message */}
            {!result.success && (
              <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                <p className="text-sm text-yellow-800 font-semibold mb-2">
                  💡 Why did authentication fail?
                </p>
                <ul className="text-sm text-yellow-700 space-y-1">
                  {!result.checks.cosine && <li>• Cosine similarity too low</li>}
                  {!result.checks.euclidean && <li>• Euclidean distance too high</li>}
                  {!result.checks.bhattacharyya && <li>• Voice distribution mismatch</li>}
                  {!result.checks.ks_test && <li>• Statistical test failed</li>}
                </ul>
                <p className="text-sm text-yellow-800 mt-3">
                  Try recording in the same environment as registration, or re-register with clearer samples.
                </p>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

export default AuthenticateTab