import type { Metadata } from "next";
import localFont from "next/font/local";
import "./globals.css";
import { FIREBASE_APP } from "@/config/FirebaseConfig";
import AnimatedBackgroundWrapper from "../components/BackgroundWrapper";
import { TopBar } from "@/components/auth/TopBar";
import Footer from "@/components/Footer";

const geistSans = localFont({
  src: "../fonts/GeistVF.woff",
  variable: "--font-geist-sans",
  weight: "100 900",
});
const geistMono = localFont({
  src: "../fonts/GeistMonoVF.woff",
  variable: "--font-geist-mono",
  weight: "100 900",
});

export const metadata: Metadata = {
  title: "Erudite Health",
  description: "Erudite Presents Molecule Generator",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        <link rel="icon" href="/favicon.ico" />
        <link
  href="https://fonts.googleapis.com/icon?family=Material+Icons"
  rel="stylesheet"
/>

      </head>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        <AnimatedBackgroundWrapper animationName="particleNetwork" />

        <TopBar />
        {children}
        <Footer />
      </body>
    </html>
  );
}
