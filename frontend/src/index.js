import React from "react";
import ReactDOM from "react-dom";
import App from "./App"; // Import the bot widget component

ReactDOM.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
  document.getElementById("root") // Mount the app to the "root" div
);