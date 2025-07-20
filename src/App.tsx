import React, { useState } from "react";
import axios from "axios";

//  Structure of each chat message (either user or bot)
interface Message {
  type: "user" | "bot"; // 'user' means our message, 'bot' means chatbot's reply
  text: string;         // actual message content
}

function App() {
  //  Stores all messages in the chat
  const [messages, setMessages] = useState<Message[]>([]);

  //  Stores current text typed in the input box
  const [input, setInput] = useState("");

  //  Function to send a message to FastAPI backend
  const sendMessage = async () => {
    if (!input.trim()) return; // avoid sending empty messages

    //  Add user message to chat immediately
    const userMsg: Message = { type: "user", text: input };
    setMessages((prev) => [...prev, userMsg]);
    setInput(""); // clear the text box after sending

    try {
      //  Send request to FastAPI backend with the question
      const res = await axios.post("http://127.0.0.1:8000/ask", {
        query: input,
      });

      //  Add bot's answer to the chat
      const botMsg: Message = {
        type: "bot",
        text: res.data.answer || "No response from backend.",
      };
      setMessages((prev) => [...prev, botMsg]);

    } catch (error) {
      console.error(" Error reaching backend:", error);

      //  Show error message in the chat if backend fails
      const errorMsg: Message = {
        type: "bot",
        text: " Error reaching backend.",
      };
      setMessages((prev) => [...prev, errorMsg]);
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 flex flex-col items-center p-6">
      {/*  Title */}
      <h1 className="text-3xl font-bold mb-6 text-blue-600">🧠 RAG Chatbot</h1>

      {/*  Chat Window */}
      <div className="w-full max-w-2xl bg-white rounded-lg shadow p-4 flex-1 mb-4 overflow-y-auto">
        {messages.map((msg, i) => (
          <div
            key={i}
            className={`p-3 rounded-xl my-2 max-w-xs ${
              msg.type === "user"
                ? "ml-auto bg-blue-500 text-white"  // user message style
                : "mr-auto bg-gray-200 text-black" // bot message style
            }`}
          >
            {msg.text}
          </div>
        ))}
      </div>

      {/*  Input Box +  Send Button */}
      <div className="w-full max-w-2xl flex">
        <input
          type="text"
          className="flex-1 p-3 rounded-l-lg border border-gray-300"
          placeholder="Ask me anything..."
          value={input}
          onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
            setInput(e.target.value) // update input state while typing
          }
          onKeyDown={(e: React.KeyboardEvent<HTMLInputElement>) => {
            if (e.key === "Enter") sendMessage(); // press Enter to send
          }}
        />
        <button
          className="bg-blue-500 text-white px-6 rounded-r-lg"
          onClick={sendMessage} // send on click
        >
          Send
        </button>
      </div>
    </div>
  );
}

export default App;
