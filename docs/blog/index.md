# Blog

<!-- This page lists all blog posts in chronological order -->

{% if blog_posts %}

{% for post in blog_posts %}

## [{{ post.title }}]({{ post.url | url }})

<p>{{ post.date.strftime('%B %d, %Y') }}
   {% if post.categories %}
   • {% for category in post.categories %}
   <a href="{{ (category.url | url) }}">{{ category }}</a>{% if not loop.last %}, {% endif %}
   {% endfor %}
   {% endif %}
</p>

{{ post.content | striptags | truncate(200) }}

[Read more]({{ post.url | url }})

---

{% endfor %}

{% endif %}
