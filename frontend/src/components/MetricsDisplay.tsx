 import { CheckCircle2, XCircle } from 'lucide-react'
import { motion } from 'framer-motion'

interface MetricsDisplayProps {
  metrics: Record<string, number>
  thresholds: Record<string, number>
  checks?: Record<string, boolean>
}

const MetricsDisplay = ({ metrics, thresholds, checks }: MetricsDisplayProps) => {
  const metricInfo: Record<string, { label: string; description: string; color: string }> = {
    cosine: {
      label: 'Cosine Similarity',
      description: 'Pattern matching (30% weight)',
      color: 'blue'
    },
    euclidean: {
      label: 'Euclidean Distance',
      description: 'Point-to-point comparison (20% weight)',
      color: 'green'
    },
    bhattacharyya: {
      label: 'Bhattacharyya Distance',
      description: 'Distribution comparison (35% weight)',
      color: 'purple'
    },
    ks_test: {
      label: 'KS-Test',
      description: 'Statistical validation (15% weight)',
      color: 'orange'
    }
  }

  const getColorClasses = (color: string, passed?: boolean) => {
    const colors: Record<string, any> = {
      blue: {
        bg: 'bg-blue-50',
        border: 'border-blue-200',
        text: 'text-blue-900',
        bar: 'bg-gradient-to-r from-blue-400 to-blue-600'
      },
      green: {
        bg: 'bg-green-50',
        border: 'border-green-200',
        text: 'text-green-900',
        bar: 'bg-gradient-to-r from-green-400 to-green-600'
      },
      purple: {
        bg: 'bg-purple-50',
        border: 'border-purple-200',
        text: 'text-purple-900',
        bar: 'bg-gradient-to-r from-purple-400 to-purple-600'
      },
      orange: {
        bg: 'bg-orange-50',
        border: 'border-orange-200',
        text: 'text-orange-900',
        bar: 'bg-gradient-to-r from-orange-400 to-orange-600'
      }
    }

    return colors[color] || colors.blue
  }

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-bold text-gray-800">Detailed Metrics</h3>
      
      <div className="grid gap-4">
        {Object.entries(metrics).map(([key, value], index) => {
          const info = metricInfo[key]
          if (!info) return null

          const threshold = thresholds[key]
          const passed = checks ? checks[key] : value >= threshold
          const colorClasses = getColorClasses(info.color, passed)
          const percentage = value * 100

          return (
            <motion.div
              key={key}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.1 }}
              className={`${colorClasses.bg} border-2 ${
                passed ? colorClasses.border : 'border-red-300'
              } rounded-lg p-4`}
            >
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <h4 className={`font-semibold ${colorClasses.text}`}>
                    {info.label}
                  </h4>
                  {passed ? (
                    <CheckCircle2 className="w-5 h-5 text-green-500" />
                  ) : (
                    <XCircle className="w-5 h-5 text-red-500" />
                  )}
                </div>
                <span className={`text-2xl font-bold ${colorClasses.text}`}>
                  {percentage.toFixed(1)}%
                </span>
              </div>

              <p className="text-sm text-gray-600 mb-3">{info.description}</p>

              {/* Progress Bar */}
              <div className="relative h-3 bg-white rounded-full overflow-hidden border border-gray-300">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${percentage}%` }}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
                  className={`h-full ${colorClasses.bar}`}
                />
                {/* Threshold Marker */}
                <div
                  className="absolute top-0 bottom-0 w-0.5 bg-red-500"
                  style={{ left: `${threshold * 100}%` }}
                  title={`Threshold: ${(threshold * 100).toFixed(1)}%`}
                >
                  <div className="absolute -top-1 left-1/2 -translate-x-1/2 w-2 h-2 bg-red-500 rounded-full" />
                </div>
              </div>

              <div className="flex justify-between items-center mt-2 text-xs text-gray-600">
                <span>
                  {passed ? '✅' : '❌'} {passed ? 'Passed' : 'Failed'}
                </span>
                <span>
                  Threshold: {(threshold * 100).toFixed(1)}%
                </span>
              </div>
            </motion.div>
          )
        })}
      </div>

      {checks && (
        <div className="bg-gray-50 rounded-lg p-4 border-2 border-gray-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Overall Result</p>
              <p className="text-lg font-bold text-gray-800">
                {Object.values(checks).filter(Boolean).length}/4 metrics passed
              </p>
            </div>
            <div className={`text-4xl ${
              Object.values(checks).filter(Boolean).length >= 3
                ? 'text-green-500'
                : 'text-red-500'
            }`}>
              {Object.values(checks).filter(Boolean).length >= 3 ? '✅' : '❌'}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default MetricsDisplay