const API_URL = 'http://localhost:8000'

class ApiError extends Error {
    //@ts-ignore
  constructor(public status: number, message: string) {
    super(message)
    this.name = 'ApiError'
  }
}

async function handleResponse(response: Response) {
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }))
    throw new ApiError(response.status, error.detail || `HTTP ${response.status}`)
  }
  return response.json()
}

export async function registerUser(username: string, audioSamples: Blob[]) {
  const formData = new FormData()
  formData.append('username', username)
  formData.append('audio1', audioSamples[0], 'sample1.wav')
  formData.append('audio2', audioSamples[1], 'sample2.wav')
  formData.append('audio3', audioSamples[2], 'sample3.wav')

  const response = await fetch(`${API_URL}/api/register`, {
    method: 'POST',
    body: formData,
  })

  return handleResponse(response)
}

export async function authenticateUser(username: string, audio: Blob) {
  const formData = new FormData()
  formData.append('username', username)
  formData.append('audio', audio, 'auth.wav')

  const response = await fetch(`${API_URL}/api/authenticate`, {
    method: 'POST',
    body: formData,
  })

  return handleResponse(response)
}

export async function getUsers() {
  const response = await fetch(`${API_URL}/api/users`)
  return handleResponse(response)
}

export async function deleteUser(username: string) {
  const response = await fetch(`${API_URL}/api/users/${username}`, {
    method: 'DELETE',
  })
  return handleResponse(response)
}

export async function getUserStats(username: string) {
  const response = await fetch(`${API_URL}/api/stats/${username}`)
  return handleResponse(response)
}

export async function checkServerHealth() {
  try {
    const response = await fetch(`${API_URL}/`)
    return response.ok
  } catch {
    return false
  }
}