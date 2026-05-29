'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
  LayoutDashboard,
  Database,
  Cpu,
  FlaskConical,
  Tag,
  Zap,
  Package,
  Video,
  FileText,
  Settings,
  SlidersHorizontal,
  ChevronLeft,
  ChevronRight,
  BookOpen,
  MapPin,
  Microscope,
  Sun,
  Moon,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { useUiStore } from '@/store/uiStore';

interface NavItem {
  href: string;
  label: string;
  icon: React.ElementType;
  badge?: string;
}

const NAV_ITEMS: NavItem[] = [
  { href: '/',           label: 'Dashboard',  icon: LayoutDashboard },
  { href: '/datasets',   label: 'Datasets',   icon: Database },
  { href: '/training',   label: 'Training',   icon: Cpu },
  { href: '/annotation', label: 'Annotation', icon: Tag },
  { href: '/inference',  label: 'Detection',  icon: Zap },
  { href: '/morphology', label: 'Morphology', icon: Microscope },
  { href: '/models',     label: 'Models',     icon: Package },
  { href: '/video',      label: 'Video',      icon: Video },
  { href: '/evaluation', label: 'Evaluation', icon: FlaskConical },
  { href: '/reports',    label: 'Reports',    icon: FileText },
  { href: '/system',     label: 'System',     icon: Settings },
  { href: '/settings',   label: 'Settings',   icon: SlidersHorizontal },
  { href: '/guide',      label: 'Guide',      icon: MapPin },
  { href: '/wiki',       label: 'Wiki',       icon: BookOpen },
];

// ---------------------------------------------------------------------------
// CTIP Logo SVG — simplified trichome + microscope emblem
// ---------------------------------------------------------------------------

function CtipLogo({ size = 28, className }: { size?: number; className?: string }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 32 32"
      fill="none"
      className={className}
      aria-label="CTIP logo"
    >
      {/* Outer ring */}
      <circle cx="16" cy="16" r="15" stroke="var(--accent)" strokeWidth="1.2" opacity="0.5" />
      {/* Inner ring */}
      <circle cx="16" cy="16" r="13" stroke="var(--accent)" strokeWidth="0.6" opacity="0.3" />

      {/* Trichome stalk */}
      <line x1="9" y1="22" x2="9" y2="13" stroke="var(--accent)" strokeWidth="1.5" strokeLinecap="round" />
      {/* Trichome bulb */}
      <circle cx="9" cy="11" r="3" stroke="var(--accent)" strokeWidth="1.2" fill="var(--accent)" fillOpacity="0.15" />
      {/* Small trichome heads on stalk */}
      <circle cx="9" cy="17" r="1" fill="var(--accent)" opacity="0.5" />
      <circle cx="9" cy="19.5" r="0.8" fill="var(--accent)" opacity="0.4" />

      {/* Microscope body */}
      <path
        d="M17 22 L17 16 M17 16 L19.5 13 M19.5 13 L19.5 10"
        stroke="var(--accent)" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round"
      />
      {/* Microscope base */}
      <path d="M14 22 L22 22" stroke="var(--accent)" strokeWidth="1.5" strokeLinecap="round" />
      {/* Microscope objective */}
      <rect x="18" y="15" width="3" height="2" rx="0.5" fill="var(--accent)" opacity="0.6" />
      {/* Microscope eyepiece */}
      <rect x="18.5" y="9.5" width="2" height="1.5" rx="0.4" fill="var(--accent)" opacity="0.7" />
      {/* Stage */}
      <path d="M14.5 19 L21 19" stroke="var(--accent)" strokeWidth="1" strokeLinecap="round" opacity="0.6" />
    </svg>
  );
}

// ---------------------------------------------------------------------------
// Sidebar
// ---------------------------------------------------------------------------

export function Sidebar() {
  const pathname = usePathname();
  const { sidebarCollapsed, toggleSidebar, theme, toggleTheme } = useUiStore();

  return (
    <aside
      className={cn(
        'flex flex-col h-full transition-all duration-200 flex-shrink-0',
        sidebarCollapsed ? 'w-14' : 'w-56',
      )}
      style={{ background: 'var(--surface)', borderRight: '1px solid var(--border)' }}
    >
      {/* Logo / Brand */}
      <div
        className="flex items-center gap-2.5 px-3 py-3 min-h-[57px] flex-shrink-0"
        style={{ borderBottom: '1px solid var(--border)' }}
      >
        <CtipLogo size={sidebarCollapsed ? 26 : 28} className="flex-shrink-0" />
        {!sidebarCollapsed && (
          <div className="min-w-0 leading-tight">
            <p
              className="text-sm font-bold tracking-tight leading-none"
              style={{ color: 'var(--text-primary)' }}
            >
              CTIP
            </p>
            <p
              className="text-[9px] font-medium uppercase tracking-widest leading-tight mt-0.5 truncate"
              style={{ color: 'var(--text-muted)' }}
            >
              Trichome Intelligence
            </p>
          </div>
        )}
      </div>

      {/* Navigation */}
      <nav className="flex-1 overflow-y-auto py-2 px-2 no-scrollbar">
        {NAV_ITEMS.map((item) => {
          const isActive =
            item.href === '/'
              ? pathname === '/'
              : pathname.startsWith(item.href);
          const Icon = item.icon;

          return (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                'flex items-center gap-2.5 px-2 py-2 rounded-md text-sm mb-0.5 transition-colors duration-100',
                sidebarCollapsed && 'justify-center',
              )}
              style={
                isActive
                  ? {
                      background: 'var(--accent-subtle)',
                      color: 'var(--accent-text)',
                      fontWeight: 600,
                    }
                  : {
                      color: 'var(--text-secondary)',
                    }
              }
              onMouseEnter={(e) => {
                if (!isActive) {
                  (e.currentTarget as HTMLElement).style.color = 'var(--text-primary)';
                  (e.currentTarget as HTMLElement).style.background = 'var(--panel)';
                }
              }}
              onMouseLeave={(e) => {
                if (!isActive) {
                  (e.currentTarget as HTMLElement).style.color = 'var(--text-secondary)';
                  (e.currentTarget as HTMLElement).style.background = 'transparent';
                }
              }}
              title={sidebarCollapsed ? item.label : undefined}
            >
              <Icon className="w-4 h-4 flex-shrink-0" />
              {!sidebarCollapsed && <span className="truncate">{item.label}</span>}
              {!sidebarCollapsed && item.badge && (
                <span
                  className="ml-auto text-xs px-1.5 py-0.5 rounded"
                  style={{
                    background: 'rgba(74,124,69,0.15)',
                    color: 'var(--accent-text)',
                  }}
                >
                  {item.badge}
                </span>
              )}
            </Link>
          );
        })}
      </nav>

      {/* Bottom controls: theme toggle + collapse */}
      <div
        className="flex-shrink-0 px-2 py-2 space-y-1"
        style={{ borderTop: '1px solid var(--border)' }}
      >
        {/* Theme toggle */}
        <button
          onClick={toggleTheme}
          className={cn(
            'flex items-center w-full px-2 py-2 rounded-md text-xs transition-colors duration-150',
            sidebarCollapsed ? 'justify-center' : 'gap-2',
          )}
          style={{ color: 'var(--text-muted)' }}
          onMouseEnter={(e) => {
            (e.currentTarget as HTMLElement).style.color = 'var(--text-secondary)';
            (e.currentTarget as HTMLElement).style.background = 'var(--panel)';
          }}
          onMouseLeave={(e) => {
            (e.currentTarget as HTMLElement).style.color = 'var(--text-muted)';
            (e.currentTarget as HTMLElement).style.background = 'transparent';
          }}
          title={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
        >
          {theme === 'dark' ? (
            <Sun className="w-4 h-4 flex-shrink-0" />
          ) : (
            <Moon className="w-4 h-4 flex-shrink-0" />
          )}
          {!sidebarCollapsed && (
            <span>{theme === 'dark' ? 'Light mode' : 'Dark mode'}</span>
          )}
        </button>

        {/* Collapse toggle */}
        <button
          onClick={toggleSidebar}
          className={cn(
            'flex items-center justify-center w-full px-2 py-2 rounded-md transition-colors duration-150',
            sidebarCollapsed ? '' : 'gap-2',
          )}
          style={{ color: 'var(--text-muted)' }}
          onMouseEnter={(e) => {
            (e.currentTarget as HTMLElement).style.color = 'var(--text-secondary)';
            (e.currentTarget as HTMLElement).style.background = 'var(--panel)';
          }}
          onMouseLeave={(e) => {
            (e.currentTarget as HTMLElement).style.color = 'var(--text-muted)';
            (e.currentTarget as HTMLElement).style.background = 'transparent';
          }}
          title={sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        >
          {sidebarCollapsed ? (
            <ChevronRight className="w-4 h-4" />
          ) : (
            <>
              <ChevronLeft className="w-4 h-4" />
              <span className="text-xs">Collapse</span>
            </>
          )}
        </button>
      </div>
    </aside>
  );
}
