---
title: ChatGPT
draft: true
tags:
foam_template:
  description: Standard blank notes with YAML frontmatter
---

## Exporting ChatGPT chat data

1. Go to https://chatgpt.com/#settings/DataControls
2. Click "Export data"
3. An email will be sent to your registered account with a link to download
   a zip file of the data
4. Extract the `conversations.json` file from the zip into `private/chatgpt`
5. Run the `parse_chatgpt.py` script
6. Copy or move any necessary transcripts from `private/chatgpt/transcripts` to `content/chatgpt`
