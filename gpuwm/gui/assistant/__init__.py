"""The assistant built into ``gpuwm gui``: say what you want to see, get a plan on the page.

A local open-weight model on the person's own card (``local``), asked
through a chat-completions server (``llm``), reads the request once and
answers every setting as a typed decision over ArWen's own tables
(``decide``, ``plan``).  Its tools are the page's own API (``agent``);
what it does shows up on the page as edits the person can change, and it
never starts, stops or deletes a run, installs anything or touches
another machine without the person's click.  Routes: ``service``.
"""
