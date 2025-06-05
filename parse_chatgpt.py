from json import loads as jloads
from pathlib import Path
from datetime import datetime, UTC
from yaml import dump as ydump

HERE = Path(__file__).parent
privatedir = HERE/'private'
contentdir = HERE/'content'

private_chatgptdir = privatedir/'chatgpt'
public_chatgptdir = contentdir/'chatgpt'
chatgptfile = private_chatgptdir/'conversations.json'
# transcriptdir = public_chatgptdir/'transcripts'
transcriptdir = private_chatgptdir/'transcripts'

transcriptdir.mkdir(parents=True, exist_ok=True)
chatgptdata = jloads(chatgptfile.read_text())


def utc_from_timestamp(ts: str | int | float):
    return datetime.fromtimestamp(float(ts), UTC) if ts else None


def process_msg(msgdata: dict):
    msg: dict = msgdata.pop('message', None)
    if msg:
        msg.update(msgdata)

        authordata: dict = msg.pop('author')
        msg['message_author'] = authordata

        metadata: dict = msg.get('metadata', {})
        author = authordata['role']
        if author == "assistant" or author == "tool":
            author = "ChatGPT"
        elif author == "system":
            if metadata.get('is_user_system_message'):
                author = "Custom user info"
            else:
                return

        if metadata.get('is_visually_hidden_from_conversation'):
            return

        parts = None
        content: dict = msg.pop('content', None)
        if content:
            parts = content.get('parts')

        if not content or not parts:
            return

        createtime = utc_from_timestamp(msg.pop('create_time'))
        lastmodtime = utc_from_timestamp(msg.pop('update_time'))

        msgstr = '---\n'
        msgstr += 'draft: true\n'
        msgstr += 'tags:\n'
        msgstr += '- ChatGPT\n'
        msgstr += 'type: ChatGPT message\n'
        msgstr += f'date: {createtime}\n'
        msgstr += f'lastmod: {lastmodtime}\n'
        msgstr += f'author: {author}\n'
        msgstr += '\n'
        msgstr += ydump(msg)
        msgstr += '---\n'
        # msgstr += f'**{author}**<br>\n*{createtime}*\n\n'
        # header = f'\n**{author}:** *({createtime.isoformat(' ', 'minutes')})*\n\n'  # noqa: E501
        header = f'\n## {author} ({createtime.isoformat(' ', 'minutes')})\n\n'  # noqa: E501

        msgtxt = ''
        if content['content_type'] in ("text", "multimodal_text"):
            for part in parts:
                if isinstance(part, str) and part:
                    msgtxt += part
                elif isinstance(part, dict):
                    content_type = part['content_type']
                    if content_type == "audio_transcription":
                        msgtxt += part['text']
                    elif content_type in ("audio_asset_pointer",
                                          "image_asset_pointer",
                                          "video_container_asset_pointer"):
                        msgtxt += str(part)
                    elif content_type == "real_time_user_audio_video_asset_pointer":  # noqa: E501
                        if 'audio_asset_pointer' in part:
                            msgtxt += str(part['audio_asset_pointer'])

                        if 'video_container_asset_pointer' in part:
                            msgtxt += str(part['video_container_asset_pointer'])  # noqa: E501

                        for frm in part['frames_asset_pointers']:
                            msgtxt += str(frm)
                else:
                    continue

                msgtxt += '\n\n'

        if msgtxt:
            return msgstr, header + msgtxt


for chat in chatgptdata:
    chat: dict
    title: str = chat.pop('title')
    createtime = utc_from_timestamp(chat.pop('create_time'))
    lastmodtime = utc_from_timestamp(chat.pop('update_time'))
    messages = chat.pop('mapping')
    lastnode: str = chat.pop('current_node')

    titleslug = title.replace(' ', '-').lower()
    fullfile = (transcriptdir/titleslug).with_suffix('.md')

    chatdir = transcriptdir/'parts'/titleslug
    chatdir.mkdir(parents=True, exist_ok=True)

    indexfile = chatdir/'index.md'
    indexstr = '---\n'
    indexstr += f'title: {title}\n'
    indexstr += 'draft: true\n'
    indexstr += 'tags:\n'
    indexstr += '- ChatGPT\n'
    indexstr += 'type: ChatGPT transcript\n'
    indexstr += f'date: {createtime}\n'
    indexstr += f'lastmod: {lastmodtime}\n'
    fullstr = indexstr + '---\n'
    if chat:
        indexstr += f'\n{ydump(chat)}'

    indexstr += '---\n'

    msglist = list()
    msgtxtlist = list()
    while lastnode:
        msg = messages[lastnode]
        msgfile = (chatdir/lastnode).with_suffix('.md')
        out = process_msg(msg)
        if out:
            msgfront, msgtext = out
            print(msgfile)
            msgfile.write_text(msgfront + msgtext, encoding='utf8')
            # print(msgtext)
            msglist.insert(0, lastnode)
            msgtxtlist.insert(0, msgtext)

        lastnode = msg['parent']

    for msg in msglist:
        indexstr += f'![[{msg}]]\n'

    for msg in msgtxtlist:
        fullstr += msg

    # print(indexstr)
    print(indexfile)
    indexfile.write_text(indexstr)
    print(fullfile)
    fullfile.write_text(fullstr, 'utf8')
