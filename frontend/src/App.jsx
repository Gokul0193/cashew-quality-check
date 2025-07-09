import { useState } from 'react'

import './App.css'
import CashewQualityCheck from './CashewQualityCheck'
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
