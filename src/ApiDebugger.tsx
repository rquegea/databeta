import React, { useState } from 'react';

interface CallDetails {
  url: string;
  method: string;
  headers: Record<string, string>;
  body: string;
}

const ApiDebugger: React.FC = () => {
  const [status, setStatus] = useState<string>('Not tested');
  const [response, setResponse] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [callDetails, setCallDetails] = useState<CallDetails>({
    url: 'http://127.0.0.1:5000/api/chat',  // Cambiado a 127.0.0.1 en lugar de localhost
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message: 'Hola' }, null, 2)
  });

  const testApi = async () => {
    setStatus('Testing...');
    setError(null);
    setResponse(null);
    
    try {
      console.log(`Sending request to: ${callDetails.url}`);
      console.log(`Method: ${callDetails.method}`);
      console.log(`Headers:`, callDetails.headers);
      console.log(`Body:`, callDetails.body);
      
      // Asegurarse de que el body es un JSON válido
      let parsedBody;
      try {
        parsedBody = JSON.parse(callDetails.body);
      } catch (e) {
        throw new Error(`Invalid JSON in body: ${(e as Error).message}`);
      }
      
      const response = await fetch(callDetails.url, {
        method: callDetails.method,
        headers: callDetails.headers,
        body: JSON.stringify(parsedBody),
        // Añadido modo para CORS:
        mode: 'cors',
        credentials: 'include'
      });
      
      console.log(`Response status: ${response.status}`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }
      
      const data = await response.json();
      console.log('Response data:', data);
      
      setResponse(data);
      setStatus('Connected successfully');
    } catch (err) {
      console.error('API test failed:', err);
      setError(err instanceof Error ? err.message : String(err));
      setStatus('Failed to connect');
    }
  };

  const handleInputChange = (field: keyof CallDetails, value: string) => {
    if (field === 'body') {
      try {
        // Try to parse as JSON to validate
        JSON.parse(value);
        setCallDetails({ ...callDetails, [field]: value });
      } catch (e) {
        // If not valid JSON, don't update
        console.error('Invalid JSON for body');
      }
    } else if (field === 'headers') {
      try {
        // Try to parse as JSON to validate
        const headers = JSON.parse(value);
        setCallDetails({ ...callDetails, [field]: headers });
      } catch (e) {
        // If not valid JSON, don't update
        console.error('Invalid JSON for headers');
      }
    } else {
      setCallDetails({ ...callDetails, [field]: value });
    }
  };

  return (
    <div style={{ padding: '20px', maxWidth: '800px', margin: '0 auto', fontFamily: 'Arial, sans-serif' }}>
      <h2>API Connection Debugger</h2>
      
      <div style={{ marginBottom: '20px' }}>
        <div style={{ marginBottom: '10px' }}>
          <label style={{ display: 'block', marginBottom: '5px' }}>URL:</label>
          <input 
            type="text" 
            value={callDetails.url} 
            onChange={(e) => handleInputChange('url', e.target.value)}
            style={{ width: '100%', padding: '8px' }}
          />
        </div>
        
        <div style={{ marginBottom: '10px' }}>
          <label style={{ display: 'block', marginBottom: '5px' }}>Method:</label>
          <select 
            value={callDetails.method}
            onChange={(e) => handleInputChange('method', e.target.value)}
            style={{ width: '100%', padding: '8px' }}
          >
            <option value="GET">GET</option>
            <option value="POST">POST</option>
            <option value="PUT">PUT</option>
            <option value="DELETE">DELETE</option>
            <option value="OPTIONS">OPTIONS</option>
          </select>
        </div>
        
        <div style={{ marginBottom: '10px' }}>
          <label style={{ display: 'block', marginBottom: '5px' }}>Headers (JSON):</label>
          <textarea
            value={JSON.stringify(callDetails.headers, null, 2)}
            onChange={(e) => handleInputChange('headers', e.target.value)}
            style={{ width: '100%', height: '80px', padding: '8px', fontFamily: 'monospace' }}
          />
        </div>
        
        <div style={{ marginBottom: '20px' }}>
          <label style={{ display: 'block', marginBottom: '5px' }}>Body:</label>
          <textarea
            value={callDetails.body}
            onChange={(e) => handleInputChange('body', e.target.value)}
            style={{ width: '100%', height: '120px', padding: '8px', fontFamily: 'monospace' }}
          />
        </div>
        
        <button 
          onClick={testApi}
          style={{
            padding: '10px 15px',
            backgroundColor: '#4CAF50',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            fontSize: '16px'
          }}
        >
          Test API Connection
        </button>
      </div>
      
      <div style={{ marginTop: '30px' }}>
        <h3>Results</h3>
        <p><strong>Status:</strong> {status}</p>
        
        {error && (
          <div style={{ 
            color: 'white', 
            backgroundColor: '#f44336', 
            padding: '15px', 
            borderRadius: '4px',
            marginTop: '10px' 
          }}>
            <strong>Error:</strong> {error}
          </div>
        )}
        
        {response && (
          <div style={{ marginTop: '15px' }}>
            <h4>Response:</h4>
            <pre style={{ 
              backgroundColor: '#f5f5f5', 
              padding: '15px', 
              borderRadius: '4px',
              overflow: 'auto',
              maxHeight: '300px'
            }}>
              {JSON.stringify(response, null, 2)}
            </pre>
          </div>
        )}
      </div>
    </div>
  );
};

export default ApiDebugger;