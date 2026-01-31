import { useState, useRef } from 'react'
import toast from 'react-hot-toast'
import { Mic, StopCircle, Loader2, CheckCircle2, AlertCircle } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import { registerUser } from '../api/voiceAuth'
import ProgressSteps from './ProgressSteps'
import MetricsDisplay from './MetricsDisplay'

const RegisterTab = () => {
  const [username, setUsername] = useState('')
  const [samples, setSamples] = useState<Blob[]>([])
  const [isRecording, setIsRecording] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [result, setResult] = useState<any>(null)
  const [currentStep, setCurrentStep] = useState(1)
  
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const audioChunksRef = useRef<Blob[]>([])

  const steps = [
    { number: 1, title: 'Enter Username', description: 'Choose a unique username' },
    { number: 2, title: 'Record Samples', description: 'Record your voice 3 times' },
    { number: 3, title: 'Register', description: 'Complete registration' },
  ]

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
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' })
        setSamples(prev => [...prev, audioBlob])
        toast.success(`Sample ${samples.length + 1}/3 recorded!`)
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
      
      toast.success('Recording started! Speak clearly...')
    } catch (error) {
      toast.error('Microphone access denied. Please grant permission.')
      console.error('Microphone error:', error)
    }
  }

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop()
      setIsRecording(false)
    }
  }

  const handleRegister = async () => {
    if (!username.trim()) {
      toast.error('Please enter a username')
      return
    }

    if (samples.length !== 3) {
      toast.error('Please record 3 voice samples')
      return
    }

    setIsLoading(true)
    const loadingToast = toast.loading('Processing registration...')

    try {
      const data = await registerUser(username, samples)
      
      toast.success('Registration successful! 🎉', { id: loadingToast })
      setResult(data)
      setCurrentStep(3)
      
      // Reset form after 5 seconds
      setTimeout(() => {
        setUsername('')
        setSamples([])
        setResult(null)
        setCurrentStep(1)
      }, 5000)
    } catch (error: any) {
      toast.error(error.message || 'Registration failed', { id: loadingToast })
      console.error('Registration error:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const handleUsernameNext = () => {
    if (!username.trim()) {
      toast.error('Please enter a username')
      return
    }
    setCurrentStep(2)
    toast.success('Great! Now record 3 voice samples')
  }

  return (
    <div className="space-y-6">
      {/* Progress Steps */}
      <ProgressSteps steps={steps} currentStep={currentStep} />

      {/* Step 1: Username */}
      <AnimatePresence mode="wait">
        {currentStep === 1 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="space-y-4"
          >
            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-2">
                Username
              </label>
              <input
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleUsernameNext()}
                placeholder="Enter a unique username"
                className="w-full px-4 py-3 border-2 border-gray-300 rounded-lg focus:border-primary-500 focus:ring-2 focus:ring-primary-200 outline-none transition-all"
                disabled={isLoading}
              />
              <p className="text-sm text-gray-500 mt-2">
                💡 Choose a username you'll remember
              </p>
            </div>

            <button
              onClick={handleUsernameNext}
              className="w-full bg-primary-600 text-white py-3 rounded-lg font-semibold hover:bg-primary-700 transition-all duration-300 hover:shadow-lg"
            >
              Next: Record Voice Samples
            </button>
          </motion.div>
        )}

        {/* Step 2: Record Samples */}
        {currentStep === 2 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="space-y-6"
          >
            {/* Recording Instructions */}
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
              <h3 className="font-semibold text-blue-900 mb-2 flex items-center gap-2">
                <AlertCircle className="w-5 h-5" />
                Recording Tips
              </h3>
              <ul className="text-sm text-blue-800 space-y-1">
                <li>• Speak clearly: "Hello, this is my voice"</li>
                <li>• Record in a quiet environment</li>
                <li>• Keep consistent volume across all 3 samples</li>
                <li>• Each recording is 3 seconds long</li>
              </ul>
            </div>

            {/* Sample Progress */}
            <div className="flex gap-4 justify-center">
              {[1, 2, 3].map((num) => (
                <div
                  key={num}
                  className={`w-20 h-20 rounded-full flex items-center justify-center font-bold text-lg transition-all ${
                    samples.length >= num
                      ? 'bg-green-500 text-white shadow-lg'
                      : samples.length === num - 1 && isRecording
                      ? 'bg-red-500 text-white animate-pulse'
                      : 'bg-gray-200 text-gray-400'
                  }`}
                >
                  {samples.length >= num ? <CheckCircle2 /> : num}
                </div>
              ))}
            </div>

            {/* Record Button */}
            <div className="flex justify-center">
              <button
                onClick={isRecording ? stopRecording : startRecording}
                disabled={samples.length >= 3 || isLoading}
                className={`w-40 h-40 rounded-full flex flex-col items-center justify-center font-bold text-white shadow-2xl transition-all duration-300 ${
                  isRecording
                    ? 'bg-red-500 animate-pulse scale-110'
                    : samples.length >= 3
                    ? 'bg-gray-400 cursor-not-allowed'
                    : 'bg-gradient-to-br from-primary-500 to-secondary-500 hover:scale-110'
                }`}
              >
                {isRecording ? (
                  <>
                    <StopCircle className="w-16 h-16 mb-2" />
                    <span className="text-sm">Stop</span>
                  </>
                ) : (
                  <>
                    <Mic className="w-16 h-16 mb-2" />
                    <span className="text-sm">
                      {samples.length >= 3 ? 'Done!' : 'Record'}
                    </span>
                  </>
                )}
              </button>
            </div>

            <p className="text-center text-gray-600">
              Samples recorded: <span className="font-bold">{samples.length}/3</span>
            </p>

            {/* Action Buttons */}
            <div className="flex gap-4">
              <button
                onClick={() => setCurrentStep(1)}
                className="flex-1 bg-gray-200 text-gray-700 py-3 rounded-lg font-semibold hover:bg-gray-300 transition-all"
                disabled={isLoading || isRecording}
              >
                Back
              </button>
              <button
                onClick={handleRegister}
                disabled={samples.length !== 3 || isLoading || isRecording}
                className="flex-1 bg-primary-600 text-white py-3 rounded-lg font-semibold hover:bg-primary-700 transition-all duration-300 hover:shadow-lg disabled:bg-gray-400 disabled:cursor-not-allowed flex items-center justify-center gap-2"
              >
                {isLoading ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    Processing...
                  </>
                ) : (
                  'Register User'
                )}
              </button>
            </div>
          </motion.div>
        )}

        {/* Step 3: Success */}
        {currentStep === 3 && result && (
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9 }}
            className="space-y-6"
          >
            <div className="bg-green-50 border-2 border-green-500 rounded-lg p-6 text-center">
              <CheckCircle2 className="w-16 h-16 text-green-500 mx-auto mb-4" />
              <h3 className="text-2xl font-bold text-green-900 mb-2">
                Registration Successful!
              </h3>
              <p className="text-green-700">
                User <span className="font-bold">{result.username}</span> has been registered
              </p>
            </div>

            <MetricsDisplay metrics={result.consistency} thresholds={result.thresholds} />

            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
              <p className="text-sm text-blue-800">
                ✅ Your voice profile has been created with calibrated thresholds
              </p>
              <p className="text-sm text-blue-800 mt-1">
                🎯 You can now authenticate using the "Authenticate" tab
              </p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

export default RegisterTab