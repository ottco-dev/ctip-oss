'use client';

import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';
import rehypeSlug from 'rehype-slug';
import { type ComponentPropsWithoutRef } from 'react';
import 'highlight.js/styles/github-dark.css';

interface WikiRendererProps {
  content: string;
}

type HeadingProps = ComponentPropsWithoutRef<'h1'>;
type AnchorProps = ComponentPropsWithoutRef<'a'>;
type CodeProps = ComponentPropsWithoutRef<'code'> & { inline?: boolean };
type TableProps = ComponentPropsWithoutRef<'table'>;
type BlockquoteProps = ComponentPropsWithoutRef<'blockquote'>;

export function WikiRenderer({ content }: WikiRendererProps) {
  return (
    <div className="wiki-content prose prose-invert max-w-none">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeHighlight, rehypeSlug]}
        components={{
          h1: ({ children, id, ...props }: HeadingProps) => (
            <h1
              id={id}
              className="text-2xl font-bold text-text-primary mt-8 mb-4 pb-2 border-b border-border scroll-mt-20"
              {...props}
            >
              {children}
            </h1>
          ),
          h2: ({ children, id, ...props }: HeadingProps) => (
            <h2
              id={id}
              className="text-xl font-semibold text-text-primary mt-8 mb-3 pb-1 border-b border-border/50 scroll-mt-20"
              {...props}
            >
              {children}
            </h2>
          ),
          h3: ({ children, id, ...props }: HeadingProps) => (
            <h3
              id={id}
              className="text-base font-semibold text-text-primary mt-6 mb-2 scroll-mt-20"
              {...props}
            >
              {children}
            </h3>
          ),
          h4: ({ children, id, ...props }: HeadingProps) => (
            <h4
              id={id}
              className="text-sm font-semibold text-text-secondary mt-4 mb-1 scroll-mt-20"
              {...props}
            >
              {children}
            </h4>
          ),
          p: ({ children, ...props }) => (
            <p className="text-text-secondary leading-relaxed mb-4 text-sm" {...props}>
              {children}
            </p>
          ),
          a: ({ children, href, ...props }: AnchorProps) => (
            <a
              href={href}
              className="text-accent hover:text-accent/80 underline underline-offset-2 transition-colors"
              target={href?.startsWith('http') ? '_blank' : undefined}
              rel={href?.startsWith('http') ? 'noopener noreferrer' : undefined}
              {...props}
            >
              {children}
            </a>
          ),
          code: ({ children, inline, className, ...props }: CodeProps) => {
            if (inline) {
              return (
                <code
                  className="bg-panel text-accent font-mono text-xs px-1.5 py-0.5 rounded"
                  {...props}
                >
                  {children}
                </code>
              );
            }
            return (
              <code className={className} {...props}>
                {children}
              </code>
            );
          },
          pre: ({ children, ...props }) => (
            <pre
              className="bg-[#0d1117] border border-border rounded-lg p-4 overflow-x-auto text-xs leading-relaxed mb-4 [&>code]:!bg-transparent [&>code]:!p-0"
              {...props}
            >
              {children}
            </pre>
          ),
          table: ({ children, ...props }: TableProps) => (
            <div className="overflow-x-auto mb-4">
              <table
                className="w-full text-sm border-collapse"
                {...props}
              >
                {children}
              </table>
            </div>
          ),
          thead: ({ children, ...props }) => (
            <thead className="bg-panel" {...props}>
              {children}
            </thead>
          ),
          th: ({ children, ...props }) => (
            <th
              className="text-left text-text-primary font-semibold px-3 py-2 border border-border text-xs"
              {...props}
            >
              {children}
            </th>
          ),
          td: ({ children, ...props }) => (
            <td
              className="px-3 py-2 border border-border text-text-secondary text-xs"
              {...props}
            >
              {children}
            </td>
          ),
          tr: ({ children, ...props }) => (
            <tr className="hover:bg-panel/50 transition-colors" {...props}>
              {children}
            </tr>
          ),
          ul: ({ children, ...props }) => (
            <ul className="list-disc list-outside ml-5 mb-4 space-y-1 text-text-secondary text-sm" {...props}>
              {children}
            </ul>
          ),
          ol: ({ children, ...props }) => (
            <ol className="list-decimal list-outside ml-5 mb-4 space-y-1 text-text-secondary text-sm" {...props}>
              {children}
            </ol>
          ),
          li: ({ children, ...props }) => (
            <li className="leading-relaxed" {...props}>
              {children}
            </li>
          ),
          blockquote: ({ children, ...props }: BlockquoteProps) => (
            <blockquote
              className="border-l-2 border-accent pl-4 my-4 text-text-secondary italic text-sm bg-panel/30 py-2 pr-3 rounded-r"
              {...props}
            >
              {children}
            </blockquote>
          ),
          hr: ({ ...props }) => (
            <hr className="border-border my-8" {...props} />
          ),
          strong: ({ children, ...props }) => (
            <strong className="text-text-primary font-semibold" {...props}>
              {children}
            </strong>
          ),
          em: ({ children, ...props }) => (
            <em className="text-text-secondary italic" {...props}>
              {children}
            </em>
          ),
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
