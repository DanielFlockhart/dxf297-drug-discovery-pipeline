"use client";

export default function Footer() {
  return (
    <footer className="py-8 text-white rounded-t-3xl">
        <div className="mt-8 text-center text-gray-400 text-sm">
          © {new Date().getFullYear()} Erudite Health, All rights reserved.
        </div>
    </footer>
  );
}
