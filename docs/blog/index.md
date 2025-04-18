# Blog

<!-- This page lists all blog posts in chronological order -->

{% if blog_posts %}

<ul>
{% for post in blog_posts %}
  <li><a href="{{ post.url }}">{{ post.title }}</a></li>
{% endfor %}
</ul>

{% endif %}
