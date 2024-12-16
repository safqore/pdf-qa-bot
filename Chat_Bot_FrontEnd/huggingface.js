(function() {
    const scriptTag = document.currentScript;
    const hfModel = scriptTag.getAttribute('data-hf-model');
    const hfToken = scriptTag.getAttribute('data-hf-token');
  
    const HF_API_URL = `https://api-inference.huggingface.co/models/${hfModel}`;
  
    window.getLLMResponse = async function(userMessage) {
      if (!hfToken) {
        console.error("Hugging Face token not provided.");
        return "No API token available for Hugging Face.";
      }
  
      const payload = {
        inputs: userMessage,
        options: { wait_for_model: true }
      };
  
      try {
        const response = await fetch(HF_API_URL, {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${hfToken}`,
            'Content-Type': 'application/json'
          },
          body: JSON.stringify(payload)
        });
  
        if (!response.ok) {
          console.error("Hugging Face API error:", response.status, response.statusText);
          return "I'm having trouble responding right now.";
        }
  
        const data = await response.json();
        const text = data && data[0] && data[0].generated_text ? data[0].generated_text : "No response.";
        return text.trim();
      } catch (err) {
        console.error("Error calling Hugging Face API:", err);
        return "An error occurred while fetching a response.";
      }
    };
  })();
  