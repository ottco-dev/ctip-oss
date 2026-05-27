'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Search, Globe, ChevronRight, BookOpen } from 'lucide-react';
import { cn } from '@/lib/utils';
import { WIKI_PAGES, getWikiPagesBySection } from '@/content/wiki/index';
import { SECTIONS, UI_LABELS } from '@/content/wiki/types';
import { WikiLangContext, type Lang } from '@/components/wiki/WikiLangContext';

// ---------------------------------------------------------------------------
// Layout
// ---------------------------------------------------------------------------

export default function WikiLayout({ children }: { children: React.ReactNode }) {
  const [lang, setLang] = useState<Lang>('en');
  const [search, setSearch] = useState('');
  const [sectionOpen, setSectionOpen] = useState<Record<string, boolean>>({
    setup: true,
    workflow: true,
    reference: true,
  });
  const pathname = usePathname();

  // Persist language preference
  useEffect(() => {
    const saved = localStorage.getItem('wiki-lang') as Lang | null;
    if (saved && ['en', 'de', 'es'].includes(saved)) setLang(saved);
  }, []);

  function handleSetLang(l: Lang) {
    setLang(l);
    localStorage.setItem('wiki-lang', l);
  }

  const labels = UI_LABELS[lang];

  // Filter pages by search
  const filteredPages = search.trim()
    ? WIKI_PAGES.filter(
        (p) =>
          p.title[lang].toLowerCase().includes(search.toLowerCase()) ||
          p.description[lang].toLowerCase().includes(search.toLowerCase()),
      )
    : null;

  return (
    <WikiLangContext.Provider value={{ lang, setLang: handleSetLang }}>
      <div className="flex h-full overflow-hidden">
        {/* Wiki sidebar */}
        <aside className="w-64 shrink-0 flex flex-col bg-surface border-r border-border overflow-y-auto">
          {/* Header */}
          <div className="px-4 py-3 border-b border-border">
            <div className="flex items-center gap-2 mb-3">
              <BookOpen className="w-4 h-4 text-accent shrink-0" />
              <span className="text-sm font-semibold text-text-primary">Wiki</span>
            </div>

            {/* Search */}
            <div className="relative">
              <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-text-muted" />
              <input
                type="text"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder={labels.search}
                className="w-full bg-background border border-border rounded pl-8 pr-3 py-1.5 text-xs text-text-primary placeholder-text-muted focus:outline-none focus:ring-1 focus:ring-accent"
              />
            </div>
          </div>

          {/* Language switcher */}
          <div className="px-4 py-2 border-b border-border">
            <div className="flex items-center gap-1">
              <Globe className="w-3 h-3 text-text-muted shrink-0" />
              {(['en', 'de', 'es'] as Lang[]).map((l) => (
                <button
                  key={l}
                  onClick={() => handleSetLang(l)}
                  className={cn(
                    'px-2 py-0.5 text-xs rounded transition-colors',
                    lang === l
                      ? 'bg-accent text-background font-semibold'
                      : 'text-text-muted hover:text-text-primary hover:bg-panel',
                  )}
                >
                  {l.toUpperCase()}
                </button>
              ))}
            </div>
          </div>

          {/* Navigation */}
          <nav className="flex-1 py-2">
            {filteredPages !== null ? (
              // Search results
              <div className="px-2">
                {filteredPages.length === 0 ? (
                  <p className="text-xs text-text-muted px-3 py-4 text-center">No results</p>
                ) : (
                  filteredPages.map((page) => (
                    <WikiNavLink
                      key={page.slug}
                      href={`/wiki/${page.slug}`}
                      icon={page.icon}
                      label={page.title[lang]}
                      active={pathname === `/wiki/${page.slug}`}
                    />
                  ))
                )}
              </div>
            ) : (
              // Sectioned navigation
              SECTIONS.map((section) => {
                const pages = getWikiPagesBySection(section.id);
                const isOpen = sectionOpen[section.id] !== false;
                return (
                  <div key={section.id} className="mb-1">
                    <button
                      onClick={() =>
                        setSectionOpen((prev) => ({ ...prev, [section.id]: !isOpen }))
                      }
                      className="w-full flex items-center justify-between px-4 py-1.5 text-[10px] font-semibold uppercase tracking-widest text-text-muted hover:text-text-secondary transition-colors"
                    >
                      <span>{section.label[lang]}</span>
                      <ChevronRight
                        className={cn(
                          'w-3 h-3 transition-transform',
                          isOpen ? 'rotate-90' : '',
                        )}
                      />
                    </button>
                    {isOpen && (
                      <div className="px-2">
                        {pages.map((page) => (
                          <WikiNavLink
                            key={page.slug}
                            href={`/wiki/${page.slug}`}
                            icon={page.icon}
                            label={page.title[lang]}
                            active={pathname === `/wiki/${page.slug}`}
                          />
                        ))}
                      </div>
                    )}
                  </div>
                );
              })
            )}
          </nav>
        </aside>

        {/* Main content */}
        <main className="flex-1 overflow-y-auto">{children}</main>
      </div>
    </WikiLangContext.Provider>
  );
}

// ---------------------------------------------------------------------------
// Nav link helper
// ---------------------------------------------------------------------------

function WikiNavLink({
  href,
  icon,
  label,
  active,
}: {
  href: string;
  icon: string;
  label: string;
  active: boolean;
}) {
  return (
    <Link
      href={href}
      className={cn(
        'flex items-center gap-2 px-3 py-1.5 rounded text-xs transition-colors w-full',
        active
          ? 'bg-accent/15 text-accent font-medium'
          : 'text-text-secondary hover:bg-panel hover:text-text-primary',
      )}
    >
      <span className="text-sm shrink-0">{icon}</span>
      <span className="truncate">{label}</span>
    </Link>
  );
}
