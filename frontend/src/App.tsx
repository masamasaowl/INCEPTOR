import { useState } from 'react'
import { Toaster } from 'react-hot-toast'
import { Mic, Users, BarChart3, Info } from 'lucide-react'
import RegisterTab from './components/RegisterTab'
import AuthenticateTab from './components/AuthenticateTab'
import UsersTab from './components/UsersTab'
import GuideModal from './components/GuideModal'

type Tab = 'register' | 'authenticate' | 'users'

function App() {
  const [activeTab, setActiveTab] = useState<Tab>('register')
  const [showGuide, setShowGuide] = useState(true)

  const tabs = [
    { id: 'register' as Tab, label: 'Register', icon: Mic },
    { id: 'authenticate' as Tab, label: 'Authenticate', icon: BarChart3 },
    { id: 'users' as Tab, label: 'Users', icon: Users },
  ]

  return (
    <div className="min-h-screen bg-gradient-to-br from-primary-500 via-primary-600 to-secondary-500 py-8 px-4">
      <Toaster 
        position="top-right"
        toastOptions={{
          duration: 4000,
          style: {
            background: '#363636',
            color: '#fff',
          },
          success: {
            duration: 3000,
            iconTheme: {
              primary: '#22c55e',
              secondary: '#fff',
            },
          },
          error: {
            duration: 5000,
            iconTheme: {
              primary: '#ef4444',
              secondary: '#fff',
            },
          },
        }}
      />

      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-8 mb-6 text-white animate-slide-down">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-4xl font-bold mb-2 flex items-center gap-3">
                <span className="text-5xl">🎙️</span>
                INCEPTOR
              </h1>
              <p className="text-white/80 text-lg">
                Voice Biometric Authentication System
              </p>
              <p className="text-white/60 text-sm mt-1">
                172 Features • 4 Metrics • Adaptive Thresholds
              </p>
            </div>
            <button
              onClick={() => setShowGuide(true)}
              className="bg-white/20 hover:bg-white/30 p-3 rounded-full transition-all duration-300 hover:scale-110"
              title="Show Guide"
            >
              <Info className="w-6 h-6" />
            </button>
          </div>
        </div>

        {/* Main Card */}
        <div className="bg-white rounded-2xl shadow-2xl overflow-hidden animate-slide-up">
          {/* Tabs */}
          <div className="flex border-b border-gray-200">
            {tabs.map((tab) => {
              const Icon = tab.icon
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex-1 px-6 py-4 font-semibold transition-all duration-300 flex items-center justify-center gap-2 ${
                    activeTab === tab.id
                      ? 'text-primary-600 border-b-4 border-primary-600 bg-primary-50'
                      : 'text-gray-500 hover:text-gray-700 hover:bg-gray-50'
                  }`}
                >
                  <Icon className="w-5 h-5" />
                  {tab.label}
                </button>
              )
            })}
          </div>

          {/* Tab Content */}
          <div className="p-8">
            {activeTab === 'register' && <RegisterTab />}
            {activeTab === 'authenticate' && <AuthenticateTab />}
            {activeTab === 'users' && <UsersTab />}
          </div>
        </div>

        {/* Footer */}
        <div className="text-center mt-6 text-white/60 text-sm">
          <p>Built with React + TypeScript + FastAPI</p>
          <p className="mt-1">Multi-Metric Verification • Calibrated Thresholds • Real-Time Processing</p>
        </div>
      </div>

      {/* Guide Modal */}
      <GuideModal isOpen={showGuide} onClose={() => setShowGuide(false)} />
    </div>
  )
}

export default App