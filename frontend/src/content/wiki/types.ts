export type Lang = 'en' | 'de' | 'es';

export interface WikiPage {
  slug: string;
  title: Record<Lang, string>;
  description: Record<Lang, string>;
  content: Record<Lang, string>;
  section: 'setup' | 'workflow' | 'reference';
  icon: string;
}

export interface WikiSection {
  id: 'setup' | 'workflow' | 'reference';
  label: Record<Lang, string>;
}

export const SECTIONS: WikiSection[] = [
  { id: 'setup',     label: { en: 'Installation & Setup',   de: 'Installation & Einrichtung', es: 'Instalación & Configuración' } },
  { id: 'workflow',  label: { en: 'Scientific Workflow',     de: 'Wissenschaftlicher Workflow', es: 'Flujo Científico' } },
  { id: 'reference', label: { en: 'Reference',               de: 'Referenz',                    es: 'Referencia' } },
];

export const UI_LABELS: Record<Lang, Record<string, string>> = {
  en: {
    search: 'Search wiki…',
    language: 'Language',
    edit_github: 'Edit on GitHub',
    back: 'Back',
    toc: 'On this page',
    note: 'Note',
    warning: 'Warning',
    tip: 'Tip',
  },
  de: {
    search: 'Wiki durchsuchen…',
    language: 'Sprache',
    edit_github: 'Auf GitHub bearbeiten',
    back: 'Zurück',
    toc: 'Auf dieser Seite',
    note: 'Hinweis',
    warning: 'Warnung',
    tip: 'Tipp',
  },
  es: {
    search: 'Buscar en el wiki…',
    language: 'Idioma',
    edit_github: 'Editar en GitHub',
    back: 'Volver',
    toc: 'En esta página',
    note: 'Nota',
    warning: 'Advertencia',
    tip: 'Consejo',
  },
};
