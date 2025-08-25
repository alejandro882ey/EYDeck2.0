from django import template

register = template.Library()

@register.filter
def progress_bar_class(value):
    if value >= 95:
        return "bg-success text-white"
    elif value >= 50:
        return "bg-warning text-white"
    else:
        return "bg-danger text-white"