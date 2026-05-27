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
  BarChart3,
  BarChart2,
  FileText,
  Settings,
  ChevronLeft,
  ChevronRight,
  Microscope,
  Tags,
  Activity,
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
  { href: '/', label: 'Dashboard', icon: LayoutDashboard },
  { href: '/datasets', label: 'Datasets', icon: Database },
  { href: '/training', label: 'Training', icon: Cpu },
  { href: '/experiments', label: 'Experiments', icon: FlaskConical },
  { href: '/annotation', label: 'Annotation', icon: Tag },
  { href: '/inference', label: 'Detection', icon: Zap },
  { href: '/morphology', label: 'Morphology', icon: Microscope },
  { href: '/models', label: 'Models', icon: Package },
  { href: '/video', label: 'Video', icon: Video },
  { href: '/analytics', label: 'Calibration', icon: BarChart2 },
  { href: '/benchmarks', label: 'Benchmarks', icon: BarChart3 },
  { href: '/reports', label: 'Reports', icon: FileText },
  { href: '/labelstudio', label: 'Label Studio', icon: Tags },
  { href: '/processes', label: 'Processes', icon: Activity },
  { href: '/system', label: 'System', icon: Settings },
];

export function Sidebar() {
  const pathname = usePathname();
  const { sidebarCollapsed, toggleSidebar } = useUiStore();

  return (
    <aside
      className={cn(
        'flex flex-col bg-surface border-r border-border h-full transition-all duration-200',
        sidebarCollapsed ? 'w-14' : 'w-56',
      )}
    >
      {/* Logo */}
      <div className="flex items-center gap-3 px-3 py-4 border-b border-border min-h-[57px]">
        <Microscope className="w-6 h-6 text-accent shrink-0" />
        {!sidebarCollapsed && (
          <span className="text-sm font-semibold text-text-primary truncate">
            TrichomeLab
          </span>
        )}
      </div>

      {/* Navigation */}
      <nav className="flex-1 overflow-y-auto py-2 px-2">
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
                'flex items-center gap-3 px-2 py-2 rounded-md text-sm mb-0.5 transition-colors duration-100',
                isActive
                  ? 'bg-accent/20 text-accent font-medium'
                  : 'text-text-secondary hover:text-text-primary hover:bg-panel',
                sidebarCollapsed && 'justify-center',
              )}
              title={sidebarCollapsed ? item.label : undefined}
            >
              <Icon className="w-4 h-4 shrink-0" />
              {!sidebarCollapsed && (
                <span className="truncate">{item.label}</span>
              )}
              {!sidebarCollapsed && item.badge && (
                <span className="ml-auto text-xs bg-status-info/20 text-status-info px-1.5 py-0.5 rounded">
                  {item.badge}
                </span>
              )}
            </Link>
          );
        })}
      </nav>

      {/* Collapse toggle */}
      <div className="px-2 py-3 border-t border-border">
        <button
          onClick={toggleSidebar}
          className={cn(
            'flex items-center justify-center w-full px-2 py-2 rounded-md',
            'text-text-muted hover:text-text-secondary hover:bg-panel transition-colors',
            sidebarCollapsed ? '' : 'gap-2',
          )}
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
