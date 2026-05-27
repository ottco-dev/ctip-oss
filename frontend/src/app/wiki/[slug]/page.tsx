'use client';

import { useParams } from 'next/navigation';
import { notFound } from 'next/navigation';
import Link from 'next/link';
import { ChevronLeft, ChevronRight, ExternalLink } from 'lucide-react';
import { getWikiPage, WIKI_PAGES } from '@/content/wiki/index';
import { WikiRenderer } from '@/components/wiki/WikiRenderer';
import { useWikiLang } from '../layout';

export default function WikiPage() {
  const { slug } = useParams<{ slug: string }>();
  const { lang } = useWikiLang();

  const page = getWikiPage(slug);
  if (!page) notFound();

  const currentIndex = WIKI_PAGES.findIndex((p) => p.slug === slug);
  const prevPage = currentIndex > 0 ? WIKI_PAGES[currentIndex - 1] : null;
  const nextPage = currentIndex < WIKI_PAGES.length - 1 ? WIKI_PAGES[currentIndex + 1] : null;

  return (
    <article className="max-w-4xl mx-auto px-8 py-8">
      {/* Page header */}
      <header className="mb-8">
        <div className="flex items-center gap-3 mb-2">
          <span className="text-3xl">{page.icon}</span>
          <h1 className="text-2xl font-bold text-text-primary">{page.title[lang]}</h1>
        </div>
        <p className="text-text-muted text-sm">{page.description[lang]}</p>
        <div className="mt-3 h-px bg-border" />
      </header>

      {/* Content */}
      <WikiRenderer content={page.content[lang]} />

      {/* Prev / Next navigation */}
      <div className="mt-16 pt-6 border-t border-border flex items-center justify-between gap-4">
        {prevPage ? (
          <Link
            href={`/wiki/${prevPage.slug}`}
            className="flex items-center gap-2 text-text-secondary hover:text-text-primary transition-colors group"
          >
            <ChevronLeft className="w-4 h-4 group-hover:-translate-x-0.5 transition-transform" />
            <div>
              <div className="text-[10px] uppercase tracking-wider text-text-muted mb-0.5">Previous</div>
              <div className="text-sm font-medium">
                {prevPage.icon} {prevPage.title[lang]}
              </div>
            </div>
          </Link>
        ) : (
          <div />
        )}

        {nextPage ? (
          <Link
            href={`/wiki/${nextPage.slug}`}
            className="flex items-center gap-2 text-text-secondary hover:text-text-primary transition-colors group text-right"
          >
            <div>
              <div className="text-[10px] uppercase tracking-wider text-text-muted mb-0.5">Next</div>
              <div className="text-sm font-medium">
                {nextPage.icon} {nextPage.title[lang]}
              </div>
            </div>
            <ChevronRight className="w-4 h-4 group-hover:translate-x-0.5 transition-transform" />
          </Link>
        ) : (
          <div />
        )}
      </div>
    </article>
  );
}
