const models = [
  {
    name: 'Bro',
    family: 'Gemma',
    base: 'gemma-3-4b-it',
    size: '4B',
    env: 'BRO_LLM_PATH',
    url: 'https://huggingface.co/leeroy-jankins/bro',
    description: 'Balanced local assistant for text generation, document workflows, semantic retrieval, prompt engineering, and data management.'
  },
  {
    name: 'Gipity',
    family: 'GPT-OSS',
    base: 'gpt-oss-20b',
    size: '21B',
    env: 'GIPITY_LLM_PATH',
    url: 'https://huggingface.co/leeroy-jankins/gipity',
    description: 'Larger reasoning-oriented local assistant for text, document, semantic, function-calling, and guarded web-context workflows.'
  },
  {
    name: 'Buddy',
    family: 'Gemma 3',
    base: 'gemma-3-270m-it',
    size: '0.3B',
    env: 'BUDDY_LLM_PATH',
    url: 'https://huggingface.co/leeroy-jankins/buddy',
    description: 'Compact local assistant for lower-resource systems and conservative document and semantic retrieval workflows.'
  },
  {
    name: 'Boo',
    family: 'Phi',
    base: 'Phi-4-mini-instruct',
    size: '3.8B',
    env: 'BOO_LLM_PATH',
    url: 'https://huggingface.co/leeroy-jankins/boo',
    description: 'Lightweight reasoning assistant with strong support for text, document, semantic, prompt, and data workflows.'
  },
  {
    name: 'Jimi',
    family: 'Gemma',
    base: 'gemma-4-E4B-it',
    size: '4B',
    env: 'JIMI_LLM_PATH',
    url: 'https://huggingface.co/leeroy-jankins/jimi',
    description: 'Multimodal-capable local assistant for text, image, audio, document, semantic, prompt, data, coding, and function-call workflows.'
  },
  {
    name: 'Leeroy',
    family: 'Llama',
    base: 'Llama-3.2-1B-Instruct',
    size: '1B',
    env: 'LEEROY_LLM_PATH',
    url: 'https://huggingface.co/leeroy-jankins/leeroy',
    description: 'Small instruction-tuned text assistant for multilingual dialogue, summarization, retrieval, and prompt workflows.'
  },
  {
    name: 'Nisty',
    family: 'Gemma',
    base: 'gemma-4-E4B-it',
    size: '4B',
    env: 'NISTY_LLM_PATH',
    url: 'https://huggingface.co/leeroy-jankins/nisty',
    description: 'Governance and document-oriented multimodal assistant for local analytical, retrieval, coding, and structured-data workflows.'
  }
];

const grid = document.getElementById('model-grid');

for (const model of models) {
  const card = document.createElement('article');
  card.className = 'model-card';
  card.innerHTML = `
    <div class="model-head">
      <div class="model-name">${model.name}</div>
      <div class="model-size">${model.size}</div>
    </div>
    <div class="model-meta">${model.family} · ${model.base}</div>
    <p>${model.description}</p>
    <div class="env">Local path variable: <code>${model.env}</code></div>
    <div class="card-actions">
      <a class="button primary" href="${model.url}" target="_blank" rel="noopener noreferrer">Get GGUF</a>
      <a class="button secondary" href="https://is-leeroy-jenkins.github.io/Loca/" target="_blank" rel="noopener noreferrer">Setup</a>
    </div>`;
  grid.appendChild(card);
}
