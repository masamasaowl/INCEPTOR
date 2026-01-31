import { useState, useEffect } from 'react'
import toast from 'react-hot-toast'
import { Trash2, RefreshCw, User, TrendingUp } from 'lucide-react'
import { motion } from 'framer-motion'
import { getUsers, deleteUser, getUserStats } from '../api/voiceAuth'

interface UserStats {
  username: string
  registered: string
  feature_count: number
  total_attempts?: number
  successful?: number
  failed?: number
  success_rate?: number
}

const UsersTab = () => {
  const [users, setUsers] = useState<UserStats[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [selectedUser, setSelectedUser] = useState<string | null>(null)
  const [userStats, setUserStats] = useState<any>(null)

  const loadUsers = async () => {
    setIsLoading(true)
    try {
      const data = await getUsers()
      setUsers(data.users)
      toast.success(`Loaded ${data.count} users`)
    } catch (error: any) {
      toast.error('Failed to load users')
      console.error(error)
    } finally {
      setIsLoading(false)
    }
  }

  const handleDelete = async (username: string) => {
    if (!window.confirm(`Delete user "${username}"? This cannot be undone.`)) {
      return
    }

    const loadingToast = toast.loading('Deleting user...')
    try {
      await deleteUser(username)
      toast.success('User deleted!', { id: loadingToast })
      loadUsers()
      if (selectedUser === username) {
        setSelectedUser(null)
        setUserStats(null)
      }
    } catch (error: any) {
      toast.error('Failed to delete user', { id: loadingToast })
      console.error(error)
    }
  }

  const handleViewStats = async (username: string) => {
    setSelectedUser(username)
    try {
      const stats = await getUserStats(username)
      setUserStats(stats)
    } catch (error: any) {
      toast.error('Failed to load statistics')
      console.error(error)
    }
  }

  useEffect(() => {
    loadUsers()
  }, [])

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-gray-800 flex items-center gap-2">
          <User className="w-6 h-6" />
          Registered Users ({users.length})
        </h2>
        <button
          onClick={loadUsers}
          disabled={isLoading}
          className="flex items-center gap-2 px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-all disabled:bg-gray-400"
        >
          <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      </div>

      {/* Users List */}
      {isLoading ? (
        <div className="text-center py-12">
          <RefreshCw className="w-12 h-12 text-primary-500 animate-spin mx-auto mb-4" />
          <p className="text-gray-600">Loading users...</p>
        </div>
      ) : users.length === 0 ? (
        <div className="text-center py-12 bg-gray-50 rounded-lg">
          <User className="w-16 h-16 text-gray-300 mx-auto mb-4" />
          <p className="text-gray-600 text-lg font-semibold">No users registered yet</p>
          <p className="text-gray-500 text-sm mt-1">
            Use the "Register" tab to add your first user
          </p>
        </div>
      ) : (
        <div className="grid gap-4">
          {users.map((user, index) => (
            <motion.div
              key={user.username}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              className={`bg-gradient-to-r from-gray-50 to-white border-2 rounded-lg p-4 hover:shadow-lg transition-all ${
                selectedUser === user.username ? 'border-primary-500' : 'border-gray-200'
              }`}
            >
              <div className="flex items-center justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-3 mb-2">
                    <div className="w-12 h-12 bg-gradient-to-br from-primary-500 to-secondary-500 rounded-full flex items-center justify-center text-white font-bold text-lg">
                      {user.username.charAt(0).toUpperCase()}
                    </div>
                    <div>
                      <h3 className="font-bold text-lg text-gray-800">
                        {user.username}
                      </h3>
                      <p className="text-sm text-gray-500">
                        Registered: {new Date(user.registered).toLocaleDateString()}
                      </p>
                    </div>
                  </div>
                  
                  <div className="flex gap-4 text-sm">
                    <span className="text-gray-600">
                      Features: <span className="font-semibold">{user.feature_count}</span>
                    </span>
                  </div>
                </div>

                <div className="flex gap-2">
                  <button
                    onClick={() => handleViewStats(user.username)}
                    className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-all flex items-center gap-2"
                  >
                    <TrendingUp className="w-4 h-4" />
                    Stats
                  </button>
                  <button
                    onClick={() => handleDelete(user.username)}
                    className="px-4 py-2 bg-red-500 text-white rounded-lg hover:bg-red-600 transition-all flex items-center gap-2"
                  >
                    <Trash2 className="w-4 h-4" />
                    Delete
                  </button>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      )}

      {/* Statistics Panel */}
      {selectedUser && userStats && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-gradient-to-br from-blue-50 to-purple-50 border-2 border-primary-300 rounded-lg p-6"
        >
          <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
            <TrendingUp className="w-5 h-5" />
            Statistics for {selectedUser}
          </h3>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-white rounded-lg p-4 border border-gray-200">
              <p className="text-sm text-gray-600 mb-1">Total Attempts</p>
              <p className="text-3xl font-bold text-primary-600">
                {userStats.total_attempts}
              </p>
            </div>
            
            <div className="bg-white rounded-lg p-4 border border-gray-200">
              <p className="text-sm text-gray-600 mb-1">Successful</p>
              <p className="text-3xl font-bold text-green-600">
                {userStats.successful}
              </p>
            </div>
            
            <div className="bg-white rounded-lg p-4 border border-gray-200">
              <p className="text-sm text-gray-600 mb-1">Failed</p>
              <p className="text-3xl font-bold text-red-600">
                {userStats.failed}
              </p>
            </div>
            
            <div className="bg-white rounded-lg p-4 border border-gray-200">
              <p className="text-sm text-gray-600 mb-1">Success Rate</p>
              <p className="text-3xl font-bold text-primary-600">
                {userStats.success_rate.toFixed(0)}%
              </p>
            </div>
          </div>

          {userStats.recent_attempts && userStats.recent_attempts.length > 0 && (
            <div className="mt-6">
              <h4 className="font-semibold text-gray-700 mb-3">Recent Attempts</h4>
              <div className="space-y-2 max-h-60 overflow-y-auto">
                {userStats.recent_attempts.slice(-5).reverse().map((attempt: any, idx: number) => (
                  <div
                    key={idx}
                    className={`flex items-center justify-between p-3 rounded-lg ${
                      attempt.success ? 'bg-green-50 border border-green-200' : 'bg-red-50 border border-red-200'
                    }`}
                  >
                    <div className="flex items-center gap-3">
                      <span className="text-2xl">
                        {attempt.success ? '✅' : '❌'}
                      </span>
                      <div>
                        <p className="text-sm font-semibold">
                          {new Date(attempt.timestamp).toLocaleString()}
                        </p>
                        <p className="text-xs text-gray-600">
                          Score: {(attempt.combined_score * 100).toFixed(1)}%
                        </p>
                      </div>
                    </div>
                    <span className={`text-sm font-semibold ${
                      attempt.success ? 'text-green-700' : 'text-red-700'
                    }`}>
                      {attempt.checks_passed}/4 checks
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </motion.div>
      )}
    </div>
  )
}

export default UsersTab