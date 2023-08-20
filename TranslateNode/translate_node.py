import re
import os,json,asyncio

from typing import List

from alibabacloud_alimt20181012.client import Client as alimt20181012Client
from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_alimt20181012 import models as alimt_20181012_models
from alibabacloud_tea_util import models as util_models
from alibabacloud_tea_util.client import Client as UtilClient


# 获取环境变量
script_path = os.path.abspath(__file__)

# 获取当前脚本文件所在目录
script_directory = os.path.dirname(script_path)
with open(script_directory + '/env_config.json', 'r') as file:
    config = json.load(file)
    for k in config:
        os.environ[k] = config[k]


@staticmethod
def create_client(
    access_key_id: str,
    access_key_secret: str,
) -> alimt20181012Client:
    """
    使用AK&SK初始化账号Client
    @param access_key_id:
    @param access_key_secret:
    @return: Client
    @throws Exception
    """
    config = open_api_models.Config(
        # 必填，您的 AccessKey ID,
        access_key_id=access_key_id,
        # 必填，您的 AccessKey Secret,
        access_key_secret=access_key_secret
    )
    # Endpoint 请参考 https://api.aliyun.com/product/alimt
    config.endpoint = f'mt.cn-hangzhou.aliyuncs.com'
    return alimt20181012Client(config)

empty_str = re.compile('^\s*$', re.I | re.M)

async def translate(
   prompt, srcTrans=None, toTrans=None
):
    if not srcTrans:
        srcTrans = 'auto'
        
    if not toTrans:
        toTrans = 'en'
    # 请确保代码运行环境设置了环境变量 ALIBABA_CLOUD_ACCESS_KEY_ID 和 ALIBABA_CLOUD_ACCESS_KEY_SECRET。
    # 工程代码泄露可能会导致 AccessKey 泄露，并威胁账号下所有资源的安全性。以下代码示例使用环境变量获取 AccessKey 的方式进行调用，仅供参考，建议使用更安全的 STS 方式，更多鉴权访问方式请参见：https://help.aliyun.com/document_detail/378659.html
    client = create_client(os.environ['ALIBABA_CLOUD_ACCESS_KEY_ID'], os.environ['ALIBABA_CLOUD_ACCESS_KEY_SECRET'])
    translate_general_request = alimt_20181012_models.TranslateGeneralRequest(
            format_type='text',
            source_language=srcTrans,
            target_language=toTrans,
            source_text=prompt,
            scene='general'
        )
    runtime = util_models.RuntimeOptions()
    # 复制代码运行请自行打印 API 的返回值
    rs = await client.translate_general_with_options_async(translate_general_request, runtime)
    data = json.loads(UtilClient.to_jsonstring(rs))
    if(data['body']['Code']=='200'):
        return data['body']['Data']['Translated']
    else:
        return ''
        

class TranslateCLIPTextEncodeNode:

    def __init__(self):
        pass    
    
    @classmethod
    def INPUT_TYPES(s):

        return {
            "required": {
                "from_translate": (['auto'], {"default": "auto"}),
                "to_translate": (['en'], {"default": "en"} ),               
                "text": ("STRING", {"multiline": True}),
                "clip": ("CLIP", )
                }
            }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "translate_text"

    CATEGORY = "AlekPet Nodes/conditioning"

    def translate_text(self, from_translate, to_translate, text, clip):
        text = asyncio.run(translate(text, from_translate, to_translate))
        tokens = clip.tokenize(text)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return ([[cond, {"pooled_output": pooled}]], )


class TranslateTextNode:

    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):

        return {
            "required": {
                "from_translate": (['auto'], {"default": "auto"}),
                "to_translate": (['en'], {"default": "en"} ),               
                "text": ("STRING", {"multiline": True}),
                }
            }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "translate_text"

    CATEGORY = "AlekPet Nodes/text"

    def translate_text(self, from_translate, to_translate, text):
        text_tranlsated = asyncio.run(translate(text, from_translate, to_translate))
        return (text_tranlsated,)
    