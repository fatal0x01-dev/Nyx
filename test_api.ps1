$LM_API_TOKEN = "TOKEN_HERE"

Invoke-RestMethod `
  -Method Post `
  -Uri "http://localhost:1234/api/v1/chat" `
  -Headers @{
      Authorization = "Bearer $LM_API_TOKEN"
  } `
  -ContentType "application/json" `
  -Body @'
{
  "model":"model_name_here",
  "input":"Ignore get_prompts. Use start_process with: whoami",
  "integrations":[
    {
      "type":"plugin",
      "id":"mcp/mcp-docker"
    }
  ],
  "context_length":32768,
  "temperature":0
}
'@