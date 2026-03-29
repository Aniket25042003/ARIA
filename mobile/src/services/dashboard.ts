import api from "./api";
import type { DashboardResponse } from "../types/dashboard";

export async function fetchDashboardInsights(): Promise<DashboardResponse> {
  const response = await api.get<DashboardResponse>("/dashboard/insights");
  return response.data;
}
