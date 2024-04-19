ABOUT = """
# Visual Storytelling (Super early demo)
This app is a visual storytelling tool that provides a visual way to explore the content of large amounts of text.

## User stories:
- I want to get to know someone quickly, build me a story - their timeline geographically and topically (how did the topics of their work change over time) — would work well with someone who makes a lot of videos or
- I want to write an article / prepare for a meeting / research a topic and I need to go through a lot of content fast - summarize it, look at different view
- I want to summarize corporate calls, transcripts and how topics have evolved over time for a given topic

## Feedback
The two demo pages have a feedback form at the bottom right. Feedback of any form is super welcome.

## About this demo
This demo currently has two text collections:
- 6 of Lex Fridman's podcast transcripts on AI (episodes with Zuck, Musk, J. Bach, Bezos, LeCun, Sam Altman)
- 8 Company Transcripts from Q4 2023

For each collection, you can see a tree-map representation of topical content in the text on the left hand side.  
Clicking on any section will filter just the text that is related to that topic.  
The idea is to explore whether the visual representation of the text can help to understand the content of the text better.  
At the bottom right there is also a working Email feedback form.  

## Limitations
- This is a very VERY early version of this tool.
- Loading will be slow, I am sorry about that.
- Please do not use this for any other purpose than to explore the concept.
"""

SAMPLE_DETAILS = {
    'lex': {
        'title': '6 Lex Fridman Podcasts about AI',
        'motivation': 'You want to understand quickly overall themes, current conversations about AI from some of the Lex Fridman podacts',
        'details': 'The podcasts are with: Joscha Bach [#392], Mark Zuckerberg [#398], Elon Musk [#400], Jeff Bezos [#405], Yann LeCun [#416], Sam Altman [#419]'
    },
    'tr8': {
        'title': '8 Company Transcripts',
        'motivation': 'You want to catch up on major corporate topics discussed in the last quarter of 2023',
        'details': 'The companies do not go great together this will be improved. The companies are: Autodesk, Salesforce, Docusign, Nordstrom, Anheuser-Busch, Kroger, Best Buy, Snowflake. '
    }
}