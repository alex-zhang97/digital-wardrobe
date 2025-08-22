import type { NextApiRequest, NextApiResponse } from 'next';
import { Tag } from '@prisma/client';
import prisma  from '../../../lib/prisma'
// Global Client

type TagConnect = {
  where: { name: string; };
  create: { name: string; };
};

export default async function handler(req: NextApiRequest, res: NextApiResponse){
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const {
      url,
      fileName,
      embedding,
      color,
      material,
      style,
      category,
      tags,    // optional array of strings
      userId   // optional, link image to a user
    } = req.body;

    if (!url || !fileName || !embedding) {
      return res.status(400).json({ error: 'url, fileName, and embedding are required' });
    }

    // Create or connect tags
    let tagConnect: TagConnect[] = [];
    if (Array.isArray(tags)) {
      tagConnect = tags.map((tagName: string) => ({
        where: { name: tagName },
        create: { name: tagName }
      }));
    }

    const image = await prisma.image.create({
      data: {
        url,
        fileName,
        embedding,
        color,
        material,
        style,
        category,
        userId,
        tags: { connectOrCreate: tagConnect }
      },
      include: {
        tags: true
      }
    });

    res.status(200).json(image);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to insert image' });
  }
}
