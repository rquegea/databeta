import React from 'react';

interface LayoutProps {
  children: React.ReactNode;
}

export function Layout({ children }: LayoutProps) {
  return (
    <div className="min-h-screen bg-gradient-to-br from-pink-100 via-[#E6F3FF] to-orange-50 relative overflow-hidden">
      {/* Animated clouds background */}
      <div className="fixed inset-0 overflow-hidden">
        <div className="cloud-animation">
          {[...Array(8)].map((_, i) => (
            <div
              key={i}
              className="absolute bg-white/40 rounded-full w-32 h-32 blur-xl"
              style={{
                left: `${Math.random() * 100}%`,
                top: `${Math.random() * 100}%`,
                animation: `float ${20 + i * 3}s linear infinite`,
                opacity: 0.4 + Math.random() * 0.3,
              }}
            />
          ))}
        </div>
      </div>
      <div className="relative z-10 min-h-screen">
        {children}
      </div>
    </div>
  );
}