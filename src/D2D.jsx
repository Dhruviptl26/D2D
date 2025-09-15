import React, { useState } from "react";
import "./DrawToDevelopment.css";

const codeTemplates = {
  react: {
    code: `import React from "react";

function Component() {
  return <h1>Hello React 🚀</h1>;
}

export default Component;`,
    extension: 'jsx'
  },
  html: {
    code: `<!DOCTYPE html>
<html>
<head>
  <title>Hello HTML</title>
</head>
<body>
  <h1>Hello HTML 🌐</h1>
</body>
</html>`,
    extension: 'html'
  },
  vue: {
    code: `<template>
  <h1>Hello Vue 💚</h1>
</template>

<script>
export default {
  name: "Component",
};
</script>`,
    extension: 'vue'
  },
  angular: {
    code: `// Angular Component
export class AppComponent {
  title = 'Hello Angular 🅰️';
  
  constructor() {
    console.log('Angular app initialized!');
  }
}`,
    extension: 'ts'
  }
};

function DrawToDevelopment() {
  const [selectedLang, setSelectedLang] = useState("react");
  const [preview, setPreview] = useState(null);
  const [copied, setCopied] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);

  const handleImage = (e) => {
    const file = e.target.files?.[0];
    if (file && file.type.startsWith('image/')) {
      const url = URL.createObjectURL(file);
      setPreview(url);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      const url = URL.createObjectURL(file);
      setPreview(url);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    // Only set dragging to false if we're leaving the drop zone entirely
    if (!e.currentTarget.contains(e.relatedTarget)) {
      setIsDragging(false);
    }
  };

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(codeTemplates[selectedLang].code);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  const handleGenerate = () => {
    setIsGenerating(true);
    // Simulate code generation
    setTimeout(() => {
      setIsGenerating(false);
      // Here you would integrate with your AI/ML service
      alert('Code generation would happen here with your AI service!');
    }, 2000);
  };

  const removeImage = () => {
    if (preview) {
      URL.revokeObjectURL(preview);
      setPreview(null);
    }
  };

  const getLangIcon = (lang) => {
    const icons = {
      react: "⚛️",
      html: "🌐",
      vue: "💚",
      angular: "🅰️"
    };
    return icons[lang] || "💻";
  };

  const currentTemplate = codeTemplates[selectedLang];

  return (
    <div className="draw-to-dev">
      <div className="container">
        {/* MacBook Bezel */}
        <div className="macbook-bezel">
          {/* MacBook Header */}
          <div className="macbook-header">
            <div className="traffic-lights">
              <div className="traffic-light red"></div>
              <div className="traffic-light yellow"></div>
              <div className="traffic-light green"></div>
            </div>
            <div className="window-title">Draw to Development</div>
            <div className="spacer"></div>
          </div>

          {/* Main Screen */}
          <div className="macbook-screen">
            {/* Header */}
            <div className="app-header">
              <h1 className="app-title">Draw to Development</h1>
              <p className="app-subtitle">Transform your sketches into beautiful code</p>
              <div className="title-divider"></div>
            </div>

            <div className="main-grid">
              {/* Left Panel - Upload Section */}
              <div className="left-panel">
                <div className="panel">
                  <div className="panel-header">
                    <span className="status-dot blue"></span>
                    Upload Design
                  </div>
                  
                  <div 
                    className={`upload-area ${isDragging ? 'dragover' : ''}`}
                    onDrop={handleDrop}
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                  >
                    {preview ? (
                      <div className="preview-container">
                        <img
                          src={preview}
                          alt="Uploaded design"
                          className="preview-image"
                        />
                        <div className="preview-overlay">
                          <button 
                            onClick={removeImage}
                            className="remove-btn"
                          >
                            Remove
                          </button>
                        </div>
                      </div>
                    ) : (
                      <div className="upload-content">
                        <div className="upload-icon">📱</div>
                        <p className="upload-text">Drop your design here</p>
                        <p className="upload-subtext">or click to browse</p>
                      </div>
                    )}
                    
                    <input
                      type="file"
                      accept="image/*"
                      onChange={handleImage}
                      className="file-input"
                    />
                  </div>
                </div>
                <div className="code-terminal">
                  {/* Terminal Header */}
                  <div className="terminal-header">
                    <div className="terminal-controls">
                      <div className="terminal-lights">
                        <div className="terminal-light red"></div>
                        <div className="terminal-light yellow"></div>
                        <div className="terminal-light green"></div>
                      </div>
                      <span className="file-name">
                        component.{currentTemplate.extension}
                      </span>
                    </div>
                    <button
                      onClick={handleCopy}
                      className={`copy-btn ${copied ? 'copied' : ''}`}
                    >
                      {copied ? '✓ Copied' : 'Copy'}
                    </button>
                  </div>
                  
                  {/* Code Block */}
                  <div className="code-content">
                    <pre className="code-block">
                      <code>{currentTemplate.code}</code>
                    </pre>
                  </div>
                </div>
              </div>

              {/* Right Panel - Code Output */}
              <div className="right-panel">
                {/* Framework Selection */}
                <div className="panel">
                  <div className="panel-header">
                    <span className="status-dot green"></span>
                    Target Framework
                  </div>
                  
                  <div className="framework-grid">
                    {Object.keys(codeTemplates).map((lang) => (
                      <button
                        key={lang}
                        onClick={() => setSelectedLang(lang)}
                        className={`framework-btn ${selectedLang === lang ? 'active' : ''}`}
                      >
                        <span className="framework-icon">{getLangIcon(lang)}</span>
                        <div className="framework-name">{lang}</div>
                      </button>
                    ))}
                  </div>
                </div>
                {/* Generate Button */}
                <button 
                  className="generate-btn" 
                  onClick={handleGenerate}
                  disabled={isGenerating}
                >
                  {isGenerating ? (
                    <>
                      <span>🔄</span>
                      <span>Generating...</span>
                    </>
                  ) : (
                    <>
                      <span>⚡</span>
                      <span>Generate Code</span>
                    </>
                  )}
                </button>
              </div>
            </div>

            {/* Status Bar */}
            <div className="status-bar">
              <div className="status-left">
                <div className="status-item">
                  <span className="status-indicator"></span>
                  <span>Ready</span>
                </div>
                <div className="status-item">
                  <span>Framework: {selectedLang}</span>
                </div>
                {preview && (
                  <div className="status-item">
                    <span>Design uploaded</span>
                  </div>
                )}
              </div>
              <div>Draw to Development v2.0</div>
            </div>
          </div>
        </div>
        
        {/* MacBook Base */}
        <div className="macbook-base"></div>
        <div className="macbook-stand"></div>
      </div>
    </div>
  );
}

export default DrawToDevelopment;