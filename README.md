### CodeT5P tuning
```shell
python codet5p.py --load codet5p-220m --type lora --target_modules ["q","v","o"] --fp16[Optional]
```

Note: codet5p-2b+ still have problem

### todo list

- [ ] 支持llama
- [ ] 指令微调
- [ ] 优化代码结构
- [ ] 支持langchain
- [ ] 强化学习

### 感谢
[CodeT5](https://github.com/salesforce/CodeT5/tree/main)

[LLaMA-Efficient-Tuning](https://github.com/hiyouga/LLaMA-Efficient-Tuning)

[NTDXYG](https://github.com/NTDXYG)