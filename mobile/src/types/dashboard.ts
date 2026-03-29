export interface DashboardStats {
  user_name: string;
  sign_mode: {
    total_sessions: number;
    total_sentences: number;
    emotion_distribution: Record<string, number>;
    language_usage: Record<string, number>;
    avg_session_duration_seconds: number;
    recent_sentences: { text: string; emotion: string }[];
  };
  guide_mode: {
    total_navigations: number;
    completed_navigations: number;
    total_obstacle_scans: number;
  };
  sos_events: number;
}

export interface DashboardInsights {
  greeting: string;
  summary: string;
  sign_insight: string;
  guide_insight: string;
  tip: string;
  streak_text: string;
}

export interface DashboardResponse {
  stats: DashboardStats;
  insights: DashboardInsights | null;
}
