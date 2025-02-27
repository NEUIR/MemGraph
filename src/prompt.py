# All prompt will be used

# Build MemGraph
gen_entity_en = """
You are an AI assistant specialized in patent analysis. Your task is to extract key technical entities from a given patent abstract. These entities should be specific technical concepts, components, or methods that are central to the patent's innovation.

Instructions:
1. Carefully read the provided patent abstract.
2. Identify and list the most important technical entities mentioned in the patent.
3. Focus on entities that are:
   - Specific to the technology described
   - Central to the patent's claims or innovative aspects
   - Likely to be useful for understanding the patent's technical field
4. Provide each entity as a concise phrase or term (typically 1-5 words).
5. List up to 10 entities, prioritizing the most significant ones.
6. Do not include general or broad categories; focus on specific technical concepts.

Output Format:
1. [Entity 1]
2. [Entity 2]
3. [Entity 3]
...

Patent Abstract:
{abs}

Please extract the key technical entities from this patent abstract.
"""

gen_entity_zh = """
你是一位专门从事专利分析的AI助手。你的任务是从给定的专利摘要中提取关键技术实体。这些实体应该是特定的技术概念、组件或方法，它们是专利创新的核心。

指示：
1. 仔细阅读提供的专利摘要。
2. 识别并列出专利中提到的最重要的技术实体。
3. 重点关注以下实体：
   - 与所描述技术特定相关的
   - 对专利权利要求或创新方面至关重要的
   - 可能有助于理解专利技术领域的
4. 将每个实体以简洁的短语或术语形式提供（通常1-5个词）。
5. 列出最多10个实体，优先考虑最重要的实体。
6. 不包括一般或广泛的类别；专注于具体的技术概念。

输出格式：
1. [实体1]
2. [实体2]
3. [实体3]
...

专利摘要：
{abs}

请从这个专利摘要中提取关键技术实体。
"""

gen_ontology_en = """
You are a patent classification expert. In the patent examination and search process, it's often necessary to compare the technical fields of multiple related patents. I will provide an original patent abstract and its related technical entities, as well as four potentially related patent abstracts (options A, B, C, D) and their respective technical entities. The original abstract represents a patent under examination, while options A-D represent potentially related existing patents found in the database.

Your task is to generate a multi-level technical classification for the original abstract and each option, referencing but not limited to the approach of the International Patent Classification (IPC) system. These classifications will be used to evaluate the technical relevance between the original patent and existing patents, helping to determine the novelty and inventiveness of the patent.

Input:

Original abstract: {abs}
Original abstract technical entities: {entity}

Option A: {abs_a}
Option A technical entities: {entity_a}

Option B: {abs_b}
Option B technical entities: {entity_b}

Option C: {abs_c}
Option C technical entities: {entity_c}

Option D: {abs_d}
Option D technical entities: {entity_d}

Please output the classification results using the following format:

Original abstract: [Major category] > [Subcategory] > [Specific class]
Option A: [Major category] > [Subcategory] > [Specific class]
Option B: [Major category] > [Subcategory] > [Specific class]
Option C: [Major category] > [Subcategory] > [Specific class]
Option D: [Major category] > [Subcategory] > [Specific class]

Notes:
1. The classification should be in three levels.
2. Reference but don't limit yourself to the IPC classification system approach, using general, intuitive terms to describe technology categories.
3. Use 1-3 words to describe each level, gradually refining from major category to specific class.
4. Classifications should reflect the patent's core technical features, application areas, and innovation focus.
5. Make full use of the provided technical entities, which contain important technical information.
6. Maintain consistency: if two patents belong to similar fields, they should be given similar classifications.
7. Only output the classification results, do not add any additional explanations.

Based on the above information, please provide accurate and concise three-level technical classifications for the given original patent abstract and each option. Before starting the classification, please carefully read all abstracts and technical entities to ensure consistency and relevance in the classification.
"""

gen_ontology_zh = """
你是一位专利分类专家。在专利审查和检索过程中，经常需要比较多个相关专利的技术领域。我会提供一个原始专利摘要及其相关技术实体，以及四个潜在相关的专利摘要（选项A、B、C、D）及其各自的技术实体。原始摘要代表一个正在审查的专利，而选项A-D代表在数据库中找到的可能相关的现有专利。

你的任务是为原始摘要和每个选项生成一个三级技术分类，参考但不限于国际专利分类(IPC)体系的思路。这些分类将用于评估原始专利与现有专利之间的技术相关性，有助于确定专利的新颖性和创造性。

输入：

原始摘要: {abs}
原始摘要技术实体: {entity}

选项A: {abs_a}
选项A技术实体: {entity_a}

选项B: {abs_b}
选项B技术实体: {entity_b}

选项C: {abs_c}
选项C技术实体: {entity_c}

选项D: {abs_d}
选项D技术实体: {entity_d}

请使用以下格式输出分类结果：

原始摘要: [大类] > [中类] > [小类]
选项A: [大类] > [中类] > [小类]
选项B: [大类] > [中类] > [小类]
选项C: [大类] > [中类] > [小类]
选项D: [大类] > [中类] > [小类]

注意事项：
1. 分类应为三级。
2. 参考但不限于IPC分类体系的思路，使用通用、直观的术语来描述技术类别。
3. 每个级别应使用1-3个词来描述，从大类到小类逐步细化。
4. 分类应反映专利的核心技术特征、应用领域和创新重点。
5. 充分利用提供的技术实体，它们包含重要的技术信息。
6. 保持一致性：如果两个专利属于相似领域，应给予相似的分类。
7. 只输出分类结果，不要添加任何额外解释。

请基于以上信息，为给定的原始专利摘要和每个选项提供准确、简洁的三级技术分类。在开始分类之前，请仔细阅读所有摘要和技术实体，以确保分类的一致性和相关性。
"""

# Inference
inference_en = """
Refer to the following patent abstract to answer the subsequent question:
{rag_passages}
The question is as follows:
{question}
The classification information for the patent abstract is as follows:
{classification}
Only choose from options A/B/C/D, without providing additional analysis
Answer:
"""

inference_zh = """
参考以下专利摘要，回答后续问题：
{rag_passages}
问题如下：
{question}
专利摘要的分类信息如下：
{classification}
只需要在A/B/C/D四个选项中做出选择，不用给出额外分析
答案：
"""