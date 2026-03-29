export type EmotionType =
  | "neutral"
  | "happy"
  | "sad"
  | "fear"
  | "angry"
  | "surprise"
  | "disgust"
  | "sos";

export interface TranscriptEntry {
  id: string;
  text: string;
  emotion: EmotionType;
  timestamp: string;
}

// WebSocket message types
export interface WSFrameMessage {
  type: "frame";
  frame: string; // base64 JPEG
  timestamp: number;
}

export interface WSClearBufferMessage {
  type: "clear_buffer";
}

export interface WSSetLanguageMessage {
  type: "set_language";
  language: string;
}

export type WSClientMessage = WSFrameMessage | WSClearBufferMessage | WSSetLanguageMessage;

export interface WSLetterMessage {
  type: "letter";
  letter: string;
  buffer: string;
}

export interface WSEmotionMessage {
  type: "emotion";
  emotion: EmotionType;
  confidence: number;
}

export interface WSSentenceMessage {
  type: "sentence";
  text: string;
  emotion: EmotionType;
  audio_url: string;
}

export interface WSSOSMessage {
  type: "sos_triggered";
  text: string;
  audio_url: string;
}

export interface WSSignsDetectedMessage {
  type: "signs_detected";
  signs: string;
  emotion: EmotionType;
}

export interface WSStatusMessage {
  type: "status";
  message: string;
}

export interface WSErrorMessage {
  type: "error";
  message: string;
}

export type WSServerMessage =
  | WSLetterMessage
  | WSEmotionMessage
  | WSSentenceMessage
  | WSSOSMessage
  | WSSignsDetectedMessage
  | WSStatusMessage
  | WSErrorMessage;
