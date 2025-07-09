import { useState } from 'react'

import './App.css'

import CashewQuality from './CashewQuality'

function App() {
  const [count, setCount] = useState(0)

  return (
    <div >
     <CashewQuality/>
    </div>
  )
}

export default App
