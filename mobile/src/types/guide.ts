export interface ObstacleWarning {
  warning: string;
  severity: "clear" | "caution" | "danger";
  timestamp: string;
}
