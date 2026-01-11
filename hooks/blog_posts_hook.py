"""
MkDocs hooks to inject blog posts data into all blog post pages.
This allows the blog-post.html template to access all posts for category navigation.
Uses caching to avoid re-reading files on every rebuild.
"""
import yaml
from pathlib import Path
from mkdocs.structure.pages import Page
from mkdocs.config.defaults import MkDocsConfig
from mkdocs.structure.nav import Navigation


# Global cache for blog posts with their actual URLs
_blog_posts_cache = []
_cache_valid = False


def on_env(env, config: MkDocsConfig, files):
    """
    Build a list of all blog posts after all files have been processed.
    The `files` object contains the final destination URLs for all pages.
    """
    global _blog_posts_cache, _cache_valid

    # Skip if cache is still valid (same build)
    if _cache_valid and _blog_posts_cache:
        return env

    _blog_posts_cache = []
    docs_dir = Path(config['docs_dir'])

    for file in files:
        if file.src_path.startswith('blog/posts/') and file.src_path.endswith('.md'):
            # Use the page object which already has parsed metadata
            if hasattr(file, 'page') and file.page:
                page = file.page
                # Get metadata from the page object if available
                if hasattr(page, 'meta') and page.meta:
                    title = page.meta.get('title', page.title or file.name)
                    categories = page.meta.get('categories', [])
                    url = page.url

                    _blog_posts_cache.append({
                        'title': title,
                        'url': url,
                        'categories': categories,
                        'src_path': file.src_path
                    })
                    continue

            # Fallback: read frontmatter from file (only if page meta not available)
            full_path = docs_dir / file.src_path
            try:
                content = full_path.read_text(encoding='utf-8')
                if content.startswith('---'):
                    # Only parse the frontmatter, not the entire content
                    end_idx = content.find('---', 3)
                    if end_idx != -1:
                        frontmatter_str = content[3:end_idx]
                        frontmatter = yaml.safe_load(frontmatter_str)
                        if frontmatter:
                            url = '/' + file.dest_path.replace('index.html', '')

                            _blog_posts_cache.append({
                                'title': frontmatter.get('title', file.name),
                                'url': url,
                                'categories': frontmatter.get('categories', []),
                                'src_path': file.src_path
                            })
            except Exception:
                continue

    # Sort by src_path (newest files first based on date prefix)
    _blog_posts_cache.sort(key=lambda x: x['src_path'], reverse=True)
    _cache_valid = True

    return env


def on_page_context(context, page: Page, config: MkDocsConfig, nav: Navigation):
    """
    Inject all blog posts into the page context so blog-post.html
    template can render category navigation with article links.
    """
    # Only process blog post pages
    if not page.file.src_path.startswith('blog/posts/'):
        return context

    # Mark current post
    current_src = page.file.src_path
    all_posts = []

    for post_data in _blog_posts_cache:
        all_posts.append({
            'title': post_data['title'],
            'url': post_data['url'],
            'categories': post_data['categories'],
            'is_current': post_data['src_path'] == current_src
        })

    # Inject posts into context
    context['all_blog_posts'] = all_posts

    return context


def on_serve(server, config, builder):
    """Reset cache when serve mode detects file changes."""
    global _cache_valid
    _cache_valid = False
    return server
