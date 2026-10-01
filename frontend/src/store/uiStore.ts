/**
 * uiStore — Zustand store for UI state (sidebar, panels, theme).
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';

type Theme = 'dark' | 'light';

interface UiState {
  sidebarCollapsed: boolean;
  activePageId: string;
  theme: Theme;

  setSidebarCollapsed: (collapsed: boolean) => void;
  toggleSidebar: () => void;
  setActivePage: (pageId: string) => void;
  setTheme: (theme: Theme) => void;
  toggleTheme: () => void;
}

export const useUiStore = create<UiState>()(
  persist(
    (set) => ({
      sidebarCollapsed: false,
      activePageId: 'dashboard',
      theme: 'light',

      setSidebarCollapsed: (collapsed) => set({ sidebarCollapsed: collapsed }),
      toggleSidebar: () => set((s) => ({ sidebarCollapsed: !s.sidebarCollapsed })),
      setActivePage: (activePageId) => set({ activePageId }),
      setTheme: (theme) => set({ theme }),
      toggleTheme: () =>
        set((s) => ({ theme: s.theme === 'dark' ? 'light' : 'dark' })),
    }),
    {
      name: 'ctip-ui-store',
      // v1: the CTIP light design became the default - move stored themes over once
      version: 1,
      migrate: (persisted) => ({
        sidebarCollapsed: Boolean((persisted as Partial<UiState> | undefined)?.sidebarCollapsed),
        theme: 'light' as Theme,
      }),
      partialize: (state) => ({
        sidebarCollapsed: state.sidebarCollapsed,
        theme: state.theme,
      }),
    },
  ),
);
