# Shared Intelligence

> Tips & war stories from the ML‑engineering trenches.

## 👋 About

I'm Slava, an MLE who documents the tricks, traps, and dopamine hits
I meet while shipping models to production.

---

## 📌 Featured

{% if blog_posts %}
{%     set pinned = blog_posts | selectattr('pin', 'equalto', true) | list %}
{%     if pinned %}

### [{{ pinned[0].title }}]({{ pinned[0].url | url }})

<p>{{ pinned[0].date.strftime('%B %d, %Y') }}</p>
{{ pinned[0].content | striptags | truncate(200) }}
{%     else %}
### [{{ blog_posts[0].title }}]({{ blog_posts[0].url | url }})
<p>{{ blog_posts[0].date.strftime('%B %d, %Y') }}</p>
{{ blog_posts[0].content | striptags | truncate(200) }}
{%     endif %}
{% endif %}

## 🆕 Latest&nbsp;(last 5)

{% if blog_posts %}
{%     for post in blog_posts[:5] %}

- [{{ post.title }}]({{ post.url }}) – {{ post.date.strftime('%B %d, %Y') }}
  {%     endfor %}
  {% endif %}
