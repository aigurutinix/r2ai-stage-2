# Third-Party Notices

FinWhale sử dụng các thư viện mã nguồn mở qua npm (giấy phép đi kèm trong `node_modules`, không phân phối lại trong repo này). Ngoài ra, một phần mã nguồn dưới đây được **sao chép/phái sinh trực tiếp** vào repo nên kèm theo giấy phép gốc theo yêu cầu của giấy phép đó.

---

## @antv/gpt-vis

`r2ai-app/components/charts/vendor.ts` chứa 4 hàm render biểu đồ (Column, Line, Pie, Waterfall) **phái sinh từ [@antv/gpt-vis](https://github.com/antvis/GPT-Vis)**, đã chỉnh sửa để bật animation, format số theo tiếng Việt và sửa nhãn pie không tràn. Bản gốc phát hành theo giấy phép MIT:

```
The MIT License (MIT)

Copyright (c) 2024 AntV

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

Biểu đồ cũng render bằng [@antv/g2](https://github.com/antvis/G2) (MIT, © AntV), dùng như dependency npm.
