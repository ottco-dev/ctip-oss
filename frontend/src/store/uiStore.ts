/**
 * uiStore — Zustand store for UI state (sidebar, panels, modals).
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface UiState {
  sidebarCollapsed: boolean;
  activePageId: string;

  setSidebarCollapsed: (collapsed: boolean) => void;
  toggleSidebar: () => void;
  setActivePage: (pageId: string) => void;
}

export const useUiStore = create<UiState>()(
  persist(
    (set) => ({
      sidebarCollapsed: false,
      activePageId: 'dashboard',

      setSidebarCollapsed: (collapsed) => set({ sidebarCollapsed: collapsed }),
      toggleSidebar: () => set((s) => ({ sidebarCollapsed: !s.sidebarCollapsed })),
      setActivePage: (activePageId) => set({ activePageId }),
    }),
    {
      name: 'trichome-ui-store',
      partialize: (state) => ({
        sidebarCollapsed: state.sidebarCollapsed,
      }),
    },
  ),
);
