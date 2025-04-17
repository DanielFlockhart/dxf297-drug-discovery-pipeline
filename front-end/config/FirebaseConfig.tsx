// Import Firebase modules
import { initializeApp } from 'firebase/app';
import { getAuth } from 'firebase/auth';
import { getFirestore } from 'firebase/firestore';
import { getStorage } from 'firebase/storage';
import { getFunctions } from 'firebase/functions';

// Firebase configuration
const firebaseConfig = {
  apiKey: process.env.NEXT_PUBLIC_FIREBASE_API_KEY,
  authDomain: process.env.NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN,
  projectId: process.env.NEXT_PUBLIC_FIREBASE_PROJECT_ID,
  storageBucket: process.env.NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET,
  messagingSenderId: process.env.NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER,
  appId: process.env.NEXT_PUBLIC_FIREBASE_APP_ID,
  measurementId: process.env.NEXT_PUBLIC_FIREBASE_MEASUREMENT_ID,
};

// Initialize Firebase app
const FIREBASE_APP = initializeApp(firebaseConfig);

// Initialize Firebase services
const FIREBASE_AUTH = getAuth(FIREBASE_APP); // No need for custom persistence in web
const FIREBASE_DB = getFirestore(FIREBASE_APP);
const FIREBASE_STORAGE = getStorage(FIREBASE_APP);
const FIREBASE_FUNCTIONS = getFunctions(FIREBASE_APP, 'europe-west1');

// Export Firebase services
export {
  FIREBASE_APP,
  FIREBASE_AUTH,
  FIREBASE_DB,
  FIREBASE_STORAGE,
  FIREBASE_FUNCTIONS,
};
