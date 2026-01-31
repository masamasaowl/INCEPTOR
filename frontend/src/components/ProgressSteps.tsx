import { CheckCircle2 } from 'lucide-react'

interface Step {
  number: number
  title: string
  description: string
}

interface ProgressStepsProps {
  steps: Step[]
  currentStep: number
}

const ProgressSteps = ({ steps, currentStep }: ProgressStepsProps) => {
  return (
    <div className="mb-8">
      <div className="flex items-center justify-between">
        {steps.map((step, index) => (
          <div key={step.number} className="flex items-center flex-1">
            {/* Step Circle */}
            <div className="flex flex-col items-center">
              <div className={`w-12 h-12 rounded-full flex items-center justify-center font-bold transition-all ${
                currentStep > step.number
                  ? 'bg-green-500 text-white'
                  : currentStep === step.number
                  ? 'bg-primary-600 text-white ring-4 ring-primary-200'
                  : 'bg-gray-200 text-gray-400'
              }`}>
                {currentStep > step.number ? (
                  <CheckCircle2 className="w-6 h-6" />
                ) : (
                  step.number
                )}
              </div>
              <div className="mt-2 text-center">
                <p className={`text-sm font-semibold ${
                  currentStep >= step.number ? 'text-gray-800' : 'text-gray-400'
                }`}>
                  {step.title}
                </p>
                <p className="text-xs text-gray-500 max-w-[100px]">
                  {step.description}
                </p>
              </div>
            </div>

            {/* Connector Line */}
            {index < steps.length - 1 && (
              <div className={`flex-1 h-1 mx-4 transition-all ${
                currentStep > step.number ? 'bg-green-500' : 'bg-gray-200'
              }`} />
            )}
          </div>
        ))}
      </div>
    </div>
  )
}

export default ProgressSteps