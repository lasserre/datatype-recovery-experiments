Sub ToggleProgressBars()
    With ActivePresentation
        ' -------------------------------------------------
        ' Colors
        ' -------------------------------------------------
        Set colorSet = New Collection
        'colorSet.Add RGB(150, 220, 248) ' Turquoise
        colorSet.Add RGB(97, 203, 244) ' Turquoise
        'colorSet.Add RGB(246, 198, 173) ' Orange
        colorSet.Add RGB(242, 170, 132) ' Orange
        colorSet.Add RGB(229, 158, 221) ' Plum
        colorSet.Add RGB(132, 226, 145) ' Green
        colorSet.Add RGB(217, 217, 217) ' Gray
        colorSet.Add RGB(247, 173, 173) ' Red
        colorSet.Add RGB(166, 202, 236) ' Darker Blue

        num_slides = .Slides.Count

        ' -------------------------------------------------
        ' Build map of section index -> # slides in section
        ' -------------------------------------------------
        Dim sectionIdxToSize
        Set sectionIdxToSize = CreateObject("Scripting.Dictionary")
        Set sectionColors = CreateObject("Scripting.Dictionary")
        sectionSize = 0
        section = 1
        For i = 1 To .Slides.Count
            If .Slides(i).sectionIndex = section Then
                If .Slides(i).Layout = ppLayoutTitle Or .Slides(i).Layout = ppLayoutSectionHeader Then
                    ' dont count it...
                Else
                    sectionSize = sectionSize + 1
                End If
            Else
                sectionIdxToSize.Add section, sectionSize
                sectionColors.Add section, colorSet(section)
                section = section + 1
                If .Slides(i).Layout = ppLayoutTitle Or .Slides(i).Layout = ppLayoutSectionHeader Then
                    sectionSize = 0
                Else
                    sectionSize = 1
                End If
            End If
        Next i
        ' add the final one
        sectionIdxToSize.Add section, sectionSize
        sectionColors.Add section, colorSet(section)

        'For Each key In sectionIdxToSize.Keys
            'Debug.Print key, sectionIdxToSize(key)
        'Next key

        ' -------------------------------------------------
        ' Find existing progress bars and remove them
        ' -------------------------------------------------
        existing = False
        Set delete_items = New Collection

        For CS_idx = 1 To num_slides
            ' current slide
            Set CS = .Slides(CS_idx)

            For i = 1 To CS.Shapes.Count
                If CS.Shapes.Item(i).Name = "progress bar" Then
                    'Debug.Print "found existing progress bar on slide " & CS_idx
                    delete_items.Add CS.Shapes.Item(i)
                    existing = True
                End If
            Next i
        Next CS_idx

        For Each x In delete_items
            x.Delete
        Next x

        If existing Then Exit Sub

        ' -------------------------------------------------
        ' Compute num slides with content
        ' -------------------------------------------------
        num_content_slides = 0
        For CS_idx = 1 To num_slides
            If .Slides(CS_idx).Layout = ppLayoutTitle Or .Slides(CS_idx).Layout = ppLayoutSectionHeader Then
            Else
                num_content_slides = num_content_slides + 1
            End If
        Next CS_idx

        Debug.Print "found " & num_content_slides & " slides with content of " & num_slides & " total"

        ' -------------------------------------------------
        ' Render progress bars
        ' -------------------------------------------------
        For CS_idx = 1 To num_slides
            ' current slide
            Set CS = .Slides(CS_idx)

            'Debug.Print .Slides(1).Layout = ppLayoutSectionHeader
            'Debug.Print .Slides(1).Layout = ppLayoutTitle

            If CS.Layout = ppLayoutTitle Or CS.Layout = ppLayoutSectionHeader Then
                GoTo NextSlide
            End If

            ' -------------------------------------------------
            ' Print previous section bars
            ' -------------------------------------------------
            'Debug.Print "section count:"; sectionIdxToSize.Count

            left_margin = 0
            right_margin = 60
            bar_height = 20
            total_width = .PageSetup.SlideWidth - (left_margin + right_margin)
            'Debug.Print "total width "; total_width

            ' -------------------------------------------------
            ' Slide dots
            ' -------------------------------------------------
            dot_size = 4
            space_from_bar = 5
            render_idx = 1  ' idx of dots we rendered - it doesn't include skipped section headers
            For i = 1 To CS_idx
                If .Slides(i).Layout = ppLayoutTitle Or .Slides(i).Layout = ppLayoutSectionHeader Then
                    GoTo KeepGoing
                End If

                If .Slides(i).sectionIndex = CS.sectionIndex Then
                    'dot_color = RGB(59, 125, 35)
                    dot_color = sectionColors(.Slides(i).sectionIndex)
                Else
                    dot_color = RGB(117, 117, 117)
                End If

                dot_color = sectionColors(.Slides(i).sectionIndex)

                start_x = left_margin + (total_width * render_idx / num_content_slides) - (total_width / num_content_slides / 2)
                Set sdot = CS.Shapes.AddShape(msoShapeOval, start_x, bar_height + space_from_bar, dot_size, dot_size)
                With sdot
                    .Fill.ForeColor.RGB = dot_color
                    .Line.ForeColor.RGB = dot_color
                    .Name = "progress bar"
                End With

                render_idx = render_idx + 1     ' don't increment if we Goto KeepGoing
KeepGoing:
            Next i

            ' -------------------------------------------------
            ' Slide x of y textbox
            ' -------------------------------------------------
            'tbox_y = bar_height + dot_size + 7
            tbox_y = 0
            Set tbox = CS.Shapes.AddTextbox(msoTextOrientationHorizontal, .PageSetup.SlideWidth * 0.94, tbox_y, 100, 20)
            With tbox
                .TextFrame.TextRange.Text = CS_idx & " of " & num_slides
                .TextFrame.TextRange.Characters.Font.Size = 12
                .TextFrame.TextRange.Characters.Font.Bold = True
                .Name = "progress bar"
            End With

            ' -------------------------------------------------
            ' Progress bars
            ' -------------------------------------------------
            start_x = left_margin

            ' choose to show all or just to current section
            max_rendered_section = sectionIdxToSize.Count
            'max_rendered_section = CS.sectionIndex

            For i = 1 To max_rendered_section
                ' print section i
                section_width = total_width * sectionIdxToSize(i) / num_content_slides
                section_name = .SectionProperties.Name(i)
                If section_name = "#" Then
                    section_name = ""   ' don't render a name for this section
                End If

                'Debug.Print section_width, sectionIdxToSize(i)

                Set pbar = CS.Shapes.AddShape(msoShapePentagon, start_x, 0, section_width, bar_height)
                With pbar
                    .TextFrame.TextRange.Text = section_name
                    .TextFrame.TextRange.Characters.Font.Color = RGB(0, 0, 0)
                    .TextFrame.TextRange.Characters.Font.Size = 14
                    .Fill.ForeColor.RGB = sectionColors(i)
                    If i > CS.sectionIndex Then
                        .Fill.Transparency = 0.9
                    End If

                    If i = CS.sectionIndex Then
                        .TextEffect.FontBold = True
                    End If

                    .Name = "progress bar"
                End With

                ' update for next section
                start_x = start_x + section_width
            Next i

NextSlide:
        Next CS_idx
        Debug.Print "Done"
    End With
End Sub

