import { useState } from 'react'

type Message = {
  id: string;
  text: string;
  sender: 'user' | 'agent';
  retries?: number;
}

function App() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [agentStatus, setAgentStatus] = useState<string | null>(null)

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMsg: Message = { id: Date.now().toString(), text: input, sender: 'user' };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setIsLoading(true);
    setAgentStatus("Inicjalizacja Agenta...");

    try {
      const response = await fetch('http://localhost:8000/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: userMsg.text }),
      });

      if (!response.ok || !response.body) throw new Error("Błąd serwera API");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        buffer += decoder.decode(value, { stream: true });

        const chunks = buffer.split('\n\n');
        buffer = chunks.pop() || '';
        
        for (const chunk of chunks) {
          if (chunk.startsWith('data: ')) {
            const dataStr = chunk.substring(6);
            try {
              const data = JSON.parse(dataStr);
              
              if (data.type === 'status') {
                setAgentStatus(data.message);
              } 
              else if (data.type === 'chunk') {
                setMessages(prev => {
                  const newMsgs = [...prev];
                  const lastMsg = newMsgs[newMsgs.length - 1];
                  
                  if (lastMsg && lastMsg.sender === 'agent' && lastMsg.id === 'streaming-agent') {
                    // Tworzymy kopię ostatniej wiadomości (niemutowalność Reacta!)
                    newMsgs[newMsgs.length - 1] = {
                      ...lastMsg,
                      text: lastMsg.text + data.text
                    };
                    return newMsgs;
                  } else {
                    // To jest pierwsza literka! Tworzymy dymek Agenta
                    const agentMsg: Message = { 
                      id: 'streaming-agent', 
                      text: data.text, 
                      sender: 'agent',
                      retries: 0
                    };
                    return [...newMsgs, agentMsg];
                  }
                });
              }
              else if (data.type === 'error') {
                throw new Error(data.message);
              }
            } catch (e) {
              console.error("Błąd parsowania JSON ze strumienia", e);
            }
          }
        }
      }

    } catch (error) {
      console.error(error);
      const errorMsg: Message = { 
        id: (Date.now() + 1).toString(), 
        text: "Przepraszam, wystąpił błąd w komunikacji z serwerem.", 
        sender: 'agent' 
      };
      setMessages(prev => [...prev, errorMsg]);
    } finally {
      setIsLoading(false);
      setAgentStatus(null);
    }
  }

  return (
    <div className="flex flex-col h-screen bg-slate-900 font-sans text-slate-200">
      {/* NAGŁÓWEK */}
      <header className="bg-slate-950 border-b border-slate-800 p-4 shadow-md z-10 flex justify-between items-center">
        <div>
          <h1 className="text-xl font-bold text-slate-100">RAG Legal Assistant ⚖️</h1>
          <p className="text-xs text-slate-400">Agentic AI for Polish Law</p>
        </div>
        <div className="flex items-center gap-2">
          <span className="relative flex h-3 w-3">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-3 w-3 bg-emerald-500"></span>
          </span>
          <span className="text-xs font-semibold text-slate-300">API Connected (SSE)</span>
        </div>
      </header>

      {/* OKNO CZATU */}
      <main className="flex-1 overflow-y-auto p-4 md:p-6 space-y-6 flex flex-col scrollbar-thin scrollbar-thumb-slate-700">
        {messages.length === 0 ? (
          <div className="m-auto text-center max-w-md text-slate-500">
            <div className="text-4xl mb-4 opacity-80">📚</div>
            <h2 className="text-lg font-semibold text-slate-300 mb-2">Jak mogę pomóc?</h2>
            <p className="text-sm">Zadaj pytanie dotyczące polskiego prawa (np. z Kodeksu Cywilnego lub Prawa Budowlanego), a Agent przeszuka zindeksowane ustawy.</p>
          </div>
        ) : (
          messages.map((msg) => (
            <div key={msg.id} className={`flex flex-col ${msg.sender === 'user' ? 'items-end' : 'items-start'}`}>
              <div 
                className={`max-w-[85%] md:max-w-[70%] p-4 rounded-2xl ${
                  msg.sender === 'user' 
                    ? 'bg-blue-600 text-white rounded-br-none shadow-md' 
                    : 'bg-slate-800 border border-slate-700 text-slate-200 rounded-bl-none shadow-sm'
                }`}
              >
                <p className="whitespace-pre-wrap text-sm md:text-base leading-relaxed">{msg.text}</p>
                
                {msg.sender === 'agent' && (
                  <div className="mt-3 pt-2 border-t border-slate-700 flex items-center justify-between text-xs text-slate-400">
                    <span>
                      {msg.retries === 0 
                        ? '✅ Sędzia zaakceptował dokumenty w 1. próbie' 
                        : `🔄 Przepisywanie zapytań (Próby: ${msg.retries})`}
                    </span>
                  </div>
                )}
              </div>
            </div>
          ))
        )}

        {/* LOADING STATE z raportowaniem strumieniowym (SSE) */}
        {isLoading && agentStatus && (
          <div className="flex items-start">
             <div className="bg-slate-800 border border-slate-700 p-4 rounded-2xl rounded-bl-none shadow-sm flex items-center gap-3">
                <div className="flex gap-1">
                    <span className="w-2 h-2 bg-blue-500 rounded-full animate-bounce [animation-delay:-0.3s]"></span>
                    <span className="w-2 h-2 bg-blue-500 rounded-full animate-bounce [animation-delay:-0.15s]"></span>
                    <span className="w-2 h-2 bg-blue-500 rounded-full animate-bounce"></span>
                </div>
                <span className="text-xs text-blue-400 font-mono tracking-wide">{agentStatus}</span>
             </div>
          </div>
        )}
      </main>

      {/* PASEK INPUTU */}
      <footer className="bg-slate-950 border-t border-slate-800 p-4">
        <div className="max-w-4xl mx-auto flex gap-3">
          <input 
            type="text" 
            className="flex-1 bg-slate-900 border border-slate-700 text-slate-100 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block p-3 outline-none placeholder-slate-500" 
            placeholder="Napisz pytanie prawne..." 
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && sendMessage()}
            disabled={isLoading}
          />
          <button 
            className="bg-blue-600 hover:bg-blue-700 disabled:bg-blue-900 disabled:text-slate-400 text-white font-medium rounded-lg text-sm px-6 py-3 transition-colors shadow-md"
            onClick={sendMessage}
            disabled={isLoading || !input.trim()}
          >
            Wyślij
          </button>
        </div>
      </footer>
    </div>
  )
}

export default App